"""
用桩 planner/researcher 节点检验我们自己的代码（审批门、路由、检查点器
接线），因此不发任何真实 LLM 调用、不需要 API key。它们在常规 CI 中运行
（不受 `eval` 标记限制）。
"""

from __future__ import annotations

import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from research_agent.config import settings
from research_agent.graph import build_graph, route_after_approval
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


# 一个桩图：planner → approval → researcher，无 LLM 调用。planner 在看到
# 人工反馈（计划被拒）时返回 ['edited!']，否则返回 ['a','b']。
def _stub_planner(state):
    feedback = state.get("plan_feedback")
    return {"subtasks": ["edited!"] if feedback else ["a", "b"], "plan_feedback": ""}


def _stub_researcher(state):
    return {"findings": []}


def _build_stub():
    g = StateGraph(ResearchState)
    g.add_node("planner", _stub_planner)
    g.add_node("approval", approval_node)
    g.add_node("researcher", _stub_researcher)
    g.add_edge(START, "planner")
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_edge("researcher", END)
    return g


# ── 直通行为（可直接调用：在 interrupt() 之前就返回）─────────────────────────
def test_gate_disabled_is_passthrough():
    """HITL 关闭时，节点绝不中断、也绝不标记拒绝。

    强制把标志关掉，而不是读环境默认值——开发者可能在 .env 里设了
    HITL_APPROVAL_ENABLED=true。"""
    original = settings.hitl_approval_enabled
    settings.hitl_approval_enabled = False
    try:
        assert approval_node({"subtasks": ["a"]}) == {"_plan_rejected": False}
    finally:
        settings.hitl_approval_enabled = original


def test_gate_skipped_on_critic_loop(hitl_on):
    """即使 HITL 开启，critic 回环的重新规划（iterations > 0）也不会再次审批。"""
    out = approval_node({"subtasks": ["a"], "iterations": 1})
    assert out == {"_plan_rejected": False}


# ── 完整的暂停/恢复机制（需要检查点器）────────────────────────────────────────
def test_approve(hitl_on):
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "approve"}}

        paused = app.invoke({"query": "q"}, config=cfg)
        assert paused["__interrupt__"][0].value["proposed_subtasks"] == ["a", "b"]

        done = app.invoke(Command(resume={"action": "approve"}), config=cfg)
        assert done["subtasks"] == ["a", "b"]  # 按拟定计划执行
        assert "__interrupt__" not in done


def test_edit(hitl_on):
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "edit"}}

        app.invoke({"query": "q"}, config=cfg)
        done = app.invoke(
            Command(resume={"action": "edit", "subtasks": ["x", "y", "z"]}), config=cfg
        )
        assert done["subtasks"] == ["x", "y", "z"]  # 人工的列表生效


def test_reject_replans_and_repauses(hitl_on):
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "reject"}}

        app.invoke({"query": "q"}, config=cfg)
        repaused = app.invoke(
            Command(resume={"action": "reject", "feedback": "too broad"}), config=cfg
        )
        # 回环到 planner（它看到了反馈）并再次暂停。
        assert repaused["__interrupt__"][0].value["proposed_subtasks"] == ["edited!"]

        done = app.invoke(Command(resume={"action": "approve"}), config=cfg)
        assert done["subtasks"] == ["edited!"]


def test_abort_ends_the_run(hitl_on):
    """abort 彻底停止图：直接路由到 END（不扇出、不再暂停），
    并留下一个终态的"已中止"报告，而不是悬而未决的暂停。"""
    with sqlite_saver(":memory:") as saver:
        app = _build_stub().compile(checkpointer=saver)
        cfg: RunnableConfig = {"configurable": {"thread_id": "abort"}}

        app.invoke({"query": "q"}, config=cfg)
        done = app.invoke(
            Command(resume={"action": "abort", "reason": "not worth it"}), config=cfg
        )
        assert "__interrupt__" not in done            # 没有暂停——已结束
        assert done["_aborted"] is True
        assert "not worth it" in done["report"]["summary"]
        assert done.get("findings", []) == []         # researcher 从未运行

        # 运行是终态的：再次调用同一个线程不会使其复活。
        state = app.get_state(cfg)
        assert state.next == ()                        # 没有待处理节点


def test_run_survives_restart(hitl_on, tmp_path):
    """暂停的运行是持久的：全新的连接可以恢复同一个线程。"""
    db = str(tmp_path / "checkpoints.db")
    cfg: RunnableConfig = {"configurable": {"thread_id": "survive"}}

    with sqlite_saver(db) as saver:  # "进程 1"：启动 + 暂停，然后关闭
        app = _build_stub().compile(checkpointer=saver)
        assert app.invoke({"query": "q"}, config=cfg).get("__interrupt__")

    with sqlite_saver(db) as saver:  # "进程 2"：对同一文件的新连接
        app = _build_stub().compile(checkpointer=saver)
        done = app.invoke(Command(resume={"action": "approve"}), config=cfg)
        assert done["subtasks"] == ["a", "b"]


# ── 真实图在两种模式下都能编译（不影响评估/CI 的回归）────────────────────────
def test_real_graph_builds_with_and_without_checkpointer():
    assert "approval" in build_graph().get_graph().nodes
    with sqlite_saver(":memory:") as saver:
        assert build_graph(checkpointer=saver).checkpointer is not None
