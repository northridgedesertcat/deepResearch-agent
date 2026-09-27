"""FastAPI 封装层的 API 测试。

与持久化/HITL 测试一样，这些测试用桩图来检验我们自己的代码（端点、
后台运行器、状态上报）——不发真实 LLM 调用、不需要 API key，
因此可以在常规 CI 中运行。

桩图用内存 SQLite 检查点器编译，并通过 ``create_api(graph=...)`` 注入，
应用绝不会打开真实的 ``checkpoints.db``。
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from langgraph.graph import START, END, StateGraph

from research_agent.api import create_api
from research_agent.config import settings
from research_agent.graph import route_after_approval
from research_agent.nodes.approval import approval_node
from research_agent.persistence import sqlite_saver
from research_agent.state import ResearchState


@pytest.fixture
def hitl_on():
    """在测试期间打开审批门，之后恢复默认（关闭）。"""
    original = settings.hitl_approval_enabled
    settings.hitl_approval_enabled = True
    yield
    settings.hitl_approval_enabled = original


def _report_node(state):
    """代替整个图：输出报告形状的结果 + 调用次数。"""
    return {
        "report": {
            "summary": f"Report for: {state['query']}",
            "sections": [{"title": "T", "content": "section one"}],
            "citations": ["https://example.com"],
        },
        "llm_calls": 3,
    }


def _build_stub(saver):
    g = StateGraph(ResearchState)
    g.add_node("run", _report_node)
    g.add_edge(START, "run")
    g.add_edge("run", END)
    return g.compile(checkpointer=saver)


def _poll_until(client, thread_id, targets, timeout=5.0):
    """后台运行的状态是异步变化的；轮询 GET 直到达到 ``targets`` 中的
    某个状态（如 {"completed", "error"} 或 {"interrupted"}）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/research/{thread_id}").json()
        if body["status"] in targets:
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {thread_id} did not reach {targets} within {timeout}s")


def _poll_until_done(client, thread_id, timeout=5.0):
    return _poll_until(client, thread_id, {"completed", "error"}, timeout)


# 一个 HITL 桩图，形状与真实图一致：planner → approval（真实的门）→
# researcher（扇出，每个子任务一个分支）→ synthesizer。无 LLM 调用。
# planner 在看到反馈（计划被拒）后返回 ['edited!']，否则返回 ['a','b']。
# researcher 写入*归约*后的 `findings` 通道（避免并行分支冲突），
# synthesizer 只写一次 `report`——这正是 route_after_approval 需要扇出
# 结构的原因。
def _hitl_planner(state):
    feedback = state.get("plan_feedback")
    return {"subtasks": ["edited!"] if feedback else ["a", "b"], "plan_feedback": ""}


def _hitl_researcher(state):
    return {"findings": []}


def _hitl_synth(state):
    return {"report": {"summary": "done: " + ",".join(state["subtasks"]), "sections": [], "citations": []}}


def _build_hitl_stub(saver):
    g = StateGraph(ResearchState)
    g.add_node("planner", _hitl_planner)
    g.add_node("approval", approval_node)
    g.add_node("researcher", _hitl_researcher)
    g.add_node("synthesizer", _hitl_synth)
    g.add_edge(START, "planner")
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile(checkpointer=saver)


def test_health():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.get("/health").json() == {"status": "ok"}


def test_research_lifecycle():
    """POST 把运行入队（202 + thread_id）；GET 最终返回其报告。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            resp = client.post("/research", json={"query": "what is RAG?"})
            assert resp.status_code == 202
            thread_id = resp.json()["thread_id"]
            assert resp.json()["status"] == "running"

            done = _poll_until_done(client, thread_id)
            assert done["status"] == "completed"
            assert done["report"]["summary"] == "Report for: what is RAG?"
            assert done["report"]["citations"] == ["https://example.com"]
            assert done["llm_calls"] == 3


def test_query_validation_rejects_empty_and_oversized():
    """请求模型限制 query 长度（成本/滥用防护）——空查询和超限查询
    都会在任何运行入队之前以 422 被拒绝。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.post("/research", json={"query": ""}).status_code == 422
            assert client.post("/research", json={"query": "x" * 2001}).status_code == 422
            # 处于边界的查询会被接受。
            assert client.post("/research", json={"query": "x" * 2000}).status_code == 202


def test_unknown_thread_is_404():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_stub(saver))) as client:
            assert client.get("/research/does-not-exist").status_code == 404


def test_failed_run_reports_error():
    """图抛异常时，运行以 status='error' 呈现，而不是挂死。"""

    def _boom(state):
        raise RuntimeError("planner exploded")

    with sqlite_saver(":memory:") as saver:
        g = StateGraph(ResearchState)
        g.add_node("run", _boom)
        g.add_edge(START, "run")
        g.add_edge("run", END)
        app = g.compile(checkpointer=saver)

        with TestClient(create_api(graph=app)) as client:
            thread_id = client.post("/research", json={"query": "x"}).json()["thread_id"]
            done = _poll_until_done(client, thread_id)
            assert done["status"] == "error"
            assert "planner exploded" in done["error"]


# ── 通过 HTTP 的人工回路 ─────────────────────────────────────────────────────
def test_hitl_pauses_then_resumes_approve(hitl_on):
    """HITL 开启时，POST 会在审批门暂停；resume(approve) 按计划执行。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]

            paused = _poll_until(client, thread_id, {"interrupted"})
            assert paused["status"] == "interrupted"
            assert paused["report"] is None
            assert paused["interrupt"]["proposed_subtasks"] == ["a", "b"]

            resume = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            assert resume.status_code == 202

            done = _poll_until_done(client, thread_id)
            assert done["status"] == "completed"
            assert done["report"]["summary"] == "done: a,b"  # 按拟定的计划执行


def test_hitl_resume_edit_uses_human_subtasks(hitl_on):
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})

            client.post(
                f"/research/{thread_id}/resume",
                json={"action": "edit", "subtasks": ["x", "y", "z"]},
            )
            done = _poll_until_done(client, thread_id)
            assert done["report"]["summary"] == "done: x,y,z"  # 人工的列表生效


def test_hitl_reject_repauses_then_approve(hitl_on):
    """reject 会重新规划并再次暂停，因此客户端需要第二次恢复。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})

            client.post(
                f"/research/{thread_id}/resume",
                json={"action": "reject", "feedback": "too broad"},
            )
            repaused = _poll_until(client, thread_id, {"interrupted"})
            # 回环到 planner（它看到了反馈）并再次暂停。
            assert repaused["interrupt"]["proposed_subtasks"] == ["edited!"]

            client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            done = _poll_until_done(client, thread_id)
            assert done["report"]["summary"] == "done: edited!"


def test_resume_rejected_when_not_interrupted(hitl_on):
    """恢复一个已完成的运行返回 409，而不是静默无操作。"""
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            thread_id = client.post("/research", json={"query": "q"}).json()["thread_id"]
            _poll_until(client, thread_id, {"interrupted"})
            client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            _poll_until_done(client, thread_id)

            again = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
            assert again.status_code == 409


def test_resume_unknown_thread_is_404():
    with sqlite_saver(":memory:") as saver:
        with TestClient(create_api(graph=_build_hitl_stub(saver))) as client:
            resp = client.post("/research/nope/resume", json={"action": "approve"})
            assert resp.status_code == 404
