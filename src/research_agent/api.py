"""封装可持久化研究图的 FastAPI 服务。

一个 HTTP 接口，让智能体成为真正可调用的服务而不是脚本。按照项目优先级，
接口面刻意保持很小：启动一次运行、轮询其结果，以及健康检查。

--------------------------------------------------------------------------------
后台执行 + 轮询（而非阻塞式请求）
--------------------------------------------------------------------------------
一次研究运行包含大量 LLM 调用，可能耗时数十秒。让 HTTP 连接保持这么久
很脆弱（代理和负载均衡器会超时）。因此：

    POST /research            -> 在后台线程启动一次运行，立即返回
                                 thread_id（HTTP 202 Accepted）
    GET  /research/{thread_id} -> 轮询状态；完成后报告从检查点器读回

--------------------------------------------------------------------------------
整个服务器共用一个检查点器（而非每个请求一个）
--------------------------------------------------------------------------------
``sqlite_saver()`` 是一个持有 SQLite 连接的上下文管理器。我们在 FastAPI
的 *lifespan* 中打开一次并在服务器整个生命周期内保持打开，然后在它之上
构建一次图。从后台线程运行图是安全的，因为 ``SqliteSaver.from_conn_string``
以 ``check_same_thread=False`` 打开连接，并用 ``threading.Lock`` 串行化每次
写入——并发运行会在该锁上排队，而不是损坏文件。（多进程部署请换成
``PostgresSaver``；见 ``research_agent/persistence.py``。图代码无需任何
改动——只换 saver。）
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.graph import build_graph
from research_agent.nodes.synthesizer import Section
from research_agent.persistence import sqlite_saver
from research_agent.state import ResearchState


# ── 请求 / 响应模型 ──────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    # 限制输入长度：图会扇出到多个 LLM worker，无上限的查询就是成本放大器
    # （也是个廉价的滥用入口）。2000 字符对研究问题来说足够宽裕；
    # min_length 会在任何工作入队之前以 422 拒绝空查询。
    query: str = Field(min_length=1, max_length=2000)


class ResearchAccepted(BaseModel):
    """POST /research 在运行入队的瞬间返回。"""

    thread_id: str
    status: str  # 这里恒为 "running"；其余状态请轮询 GET


class Report(BaseModel):
    """synthesizer/护栏写入 ``state['report']`` 的结构。"""

    summary: str
    sections: list[Section]
    citations: list[str]


class ResearchStatus(BaseModel):
    """GET /research/{thread_id} 返回。"""

    thread_id: str
    status: str  # running | completed | error | interrupted
    report: Report | None = None
    guardrail_flags: list[str] = []
    llm_calls: int | None = None
    error: str | None = None
    # 当 status == "interrupted" 时，为图暴露给人工的载荷
    # （拟定的计划 + 决策 schema）。其余情况为 None。
    interrupt: dict | None = None


class ResumeDecision(BaseModel):
    """POST /research/{thread_id}/resume 的请求体——人工对计划的裁决。

    与审批节点从 ``Command(resume=...)`` 读取的内容对应：
        approve  -> {"action": "approve"}
        edit     -> {"action": "edit", "subtasks": ["...", "..."]}
        reject   -> {"action": "reject", "feedback": "..."}
        abort    -> {"action": "abort", "reason": "..."}
    """

    action: str
    subtasks: list[str] | None = None
    feedback: str | None = None
    reason: str | None = None


# ── 内存中的任务注册表 ──────────────────────────────────────────────────────
# 检查点器是*结果*的唯一事实来源，但它无法告诉我们一次运行是仍在执行
# 还是中途崩溃了（一个只跑了一半的运行看起来就像一个还没有报告的检查点）。
# 这个小型线程安全注册表跟踪*本进程*启动的运行的实际状态。它刻意是
# 临时的：重启后，GET 会退回到从持久化检查点推断状态（见 _read_status）。
class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def start(self, thread_id: str) -> None:
        # 恢复运行时也用它把 interrupted 状态翻转回 running，
        # 因此这里同时重置 interrupt 载荷。
        with self._lock:
            self._jobs[thread_id] = {"status": "running", "error": None, "interrupt": None}

    def finish(self, thread_id: str) -> None:
        with self._lock:
            if thread_id in self._jobs:
                self._jobs[thread_id].update(status="completed", interrupt=None)

    def interrupt(self, thread_id: str, payload: dict) -> None:
        """记录运行已在人在回路审批门处暂停。"""
        with self._lock:
            if thread_id in self._jobs:
                self._jobs[thread_id].update(status="interrupted", interrupt=payload)

    def fail(self, thread_id: str, error: str) -> None:
        with self._lock:
            if thread_id in self._jobs:
                self._jobs[thread_id].update(status="error", error=error)

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(thread_id)
            return dict(job) if job else None


# ── 辅助函数 ─────────────────────────────────────────────────────────────────
def _run_config(thread_id: str) -> RunnableConfig:
    """构建每次运行的配置。``thread_id`` 是运行可寻址、可恢复的关键；
    metadata 与 graph.py 的演示保持一致，便于追踪标注。"""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "deep-research",
        "metadata": {
            "model_planner": settings.model_planner,
            "model_researcher": settings.model_researcher,
            "model_synthesizer": settings.model_synthesizer,
            "model_critic": settings.model_critic,
        },
    }


def _run_graph(graph, registry: JobRegistry, thread_id: str, inputs) -> None:
    """后台任务：驱动图向前运行，同时记录状态。

    ``inputs`` 在全新运行时是初始状态（``{"query": ...}``）；在继续一个
    暂停于 HITL 审批门的运行时是 ``Command(resume=decision)``——
    图对两者的处理方式相同。

    运行在线程池 worker 中。三种结局：
      * 图在 ``interrupt()`` 处暂停 —— invoke 会返回（不抛异常也不阻塞），
        结果带 ``__interrupt__`` 载荷 → 记录 "interrupted"；
      * 运行到结束 → "completed"；
      * 抛出异常 → "error"（捕获异常，运行绝不无声消失）。

    图在运行过程中会把报告写入检查点器，因此我们不返回任何东西——
    GET 会从那里把结果读回来。
    """
    try:
        result = graph.invoke(inputs, config=_run_config(thread_id))
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            registry.interrupt(thread_id, interrupts[0].value)
        else:
            registry.finish(thread_id)
    except Exception as exc:  # noqa: BLE001 — 把任何失败都暴露给调用方
        registry.fail(thread_id, f"{type(exc).__name__}: {exc}")


def _pending_interrupt(snapshot) -> dict | None:
    """尽力从检查点中提取待处理的 interrupt 载荷。

    仅用于重启之后的路径：此时活动的 JobRegistry 并不知道上一个进程
    启动的运行。防御性实现：快照内部结构可能随 LangGraph 版本变化，
    失败就返回 None。
    """
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0].value
    return None


def _read_status(graph, registry: JobRegistry, thread_id: str) -> ResearchStatus:
    """由实时跟踪 + 持久化检查点拼装出一个状态响应。"""
    snapshot = graph.get_state(_run_config(thread_id))
    values = snapshot.values or {}
    job = registry.get(thread_id)

    if job is not None:
        status, error = job["status"], job["error"]
        interrupt_payload = job.get("interrupt")
    elif values:
        # 本进程未跟踪（如重启前启动的运行）：从检查点推断。
        # 有待处理节点说明在等待（暂停于 HITL 审批门）；
        # 无待处理节点 + 有报告说明已完成。
        if snapshot.next == () and values.get("report"):
            status, interrupt_payload = "completed", None
        else:
            status, interrupt_payload = "interrupted", _pending_interrupt(snapshot)
        error = None
    else:
        # 该 id 既没有活动任务也没有检查点——它从未在此存在过。
        raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")

    report = values.get("report")
    return ResearchStatus(
        thread_id=thread_id,
        status=status,
        report=Report(**report) if report else None,
        guardrail_flags=values.get("guardrail_flags", []),
        llm_calls=values.get("llm_calls"),
        error=error,
        interrupt=interrupt_payload,
    )


# ── 应用工厂 ─────────────────────────────────────────────────────────────────
def create_api(graph=None) -> FastAPI:
    """构建 FastAPI 应用。

    ``graph`` 允许测试注入一个桩图（已用内存检查点器编译完成），使测试
    套件不发起任何 LLM 调用、不触碰真实数据库文件。
    传 ``None``（生产/开发）时，lifespan 会打开 SQLite 检查点器并在其上
    构建真实的图。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 每服务器共享的资源：启动时创建一次，退出时销毁。
        app.state.registry = JobRegistry()
        app.state.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="research")
        try:
            if graph is not None:
                app.state.graph = graph
                yield
            else:
                # 检查点器连接在整个服务器生命周期内保持打开。
                with sqlite_saver() as saver:
                    app.state.graph = build_graph(checkpointer=saver)
                    yield
        finally:
            app.state.executor.shutdown(wait=False, cancel_futures=True)

    api = FastAPI(
        title="Deep Research Agent",
        version="0.1.0",
        summary="Durable, citation-backed research agent exposed over HTTP.",
        lifespan=lifespan,
    )

    @api.get("/health")
    def health() -> dict:
        """供托管平台使用的存活探针。"""
        return {"status": "ok"}

    @api.post("/research", response_model=ResearchAccepted, status_code=202)
    def start_research(req: ResearchRequest, request: Request) -> ResearchAccepted:
        """把一次研究运行入队，并立即返回其 thread_id。

        202 Accepted = "已受理，尚未完成" —— 结果请轮询 GET。
        """
        thread_id = uuid.uuid4().hex
        request.app.state.registry.start(thread_id)
        request.app.state.executor.submit(
            _run_graph,
            request.app.state.graph,
            request.app.state.registry,
            thread_id,
            cast(ResearchState, {"query": req.query}),
        )
        return ResearchAccepted(thread_id=thread_id, status="running")

    @api.get("/research/{thread_id}", response_model=ResearchStatus)
    def get_research(thread_id: str, request: Request) -> ResearchStatus:
        """获取一次运行的状态，完成后连同报告一起返回。

        当 ``status == "interrupted"`` 时，响应携带 ``interrupt``
        载荷（拟定的计划）——请把它 POST 回 resume 端点。
        """
        return _read_status(request.app.state.graph, request.app.state.registry, thread_id)

    @api.post("/research/{thread_id}/resume", response_model=ResearchAccepted, status_code=202)
    def resume_research(thread_id: str, decision: ResumeDecision, request: Request) -> ResearchAccepted:
        """带着人工的裁决，恢复暂停于人在回路审批门的运行。

        仅当运行当前处于 ``interrupted`` 状态时有效。与 POST /research 一样，
        这里返回 202 并在后台继续运行图——``reject`` 会重新规划并*再次*
        暂停，因此客户端可能需要多次恢复（每次之间轮询 GET）。
        """
        graph, registry = request.app.state.graph, request.app.state.registry

        # 运行必须存在、且确实在等待输入。先查实时注册表；
        # 查不到再退回持久化检查点（重启后的情况）。
        job = registry.get(thread_id)
        if job is not None:
            if job["status"] != "interrupted":
                raise HTTPException(
                    status_code=409,
                    detail=f"Run is '{job['status']}', not waiting for input.",
                )
        else:
            snapshot = graph.get_state(_run_config(thread_id))
            if not snapshot.values:
                raise HTTPException(status_code=404, detail=f"Unknown thread_id: {thread_id}")
            if snapshot.next == ():
                raise HTTPException(status_code=409, detail="Run is finished, not waiting for input.")

        # 翻转回 running，并通过 Command(resume=...) 把决策喂给图。
        # exclude_none 会去掉与本操作无关的字段（如 approve 时没有
        # 'subtasks'），与审批节点读取的内容保持一致。
        registry.start(thread_id)
        request.app.state.executor.submit(
            _run_graph, graph, registry, thread_id, Command(resume=decision.model_dump(exclude_none=True))
        )
        return ResearchAccepted(thread_id=thread_id, status="running")

    return api


# 供 `uvicorn research_agent.api:api` 使用的模块级应用。
api = create_api()
