from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig
from research_agent.config import settings
from research_agent.guardrails import RETRY_POLICY
from research_agent.state import ResearchState
from research_agent.nodes.planner import planner_node
from research_agent.nodes.approval import approval_node
from research_agent.nodes.researcher import researcher_node
from research_agent.nodes.critic import critic_node
from research_agent.nodes.synthesizer import synthesizer_node
from research_agent.nodes.guardrails import input_guard_node, output_guard_node
from research_agent.progress import progress
from pathlib import Path
from typing import cast
import sys

def route_after_input_guard(state) -> str:
    """输入护栏：在任何开销产生之前，拒绝超范围/不安全的查询。"""
    return END if state.get("rejected") else "planner"

def route_to_researchers(state) -> list[Send]:
    """每个子任务发出一个 Send → N 条并行的 researcher 分支。"""
    sends = [
        Send("researcher", {"subtasks": [task], "findings": []})
        for task in state["subtasks"]
    ]
    progress(f"🔀 扇出 {len(sends)} 个并行研究员分支")
    return sends

def route_after_approval(state):
    """人在回路审批门之后：中止、重新起草，或扇出。

    条件边可以返回节点名（字符串）、END 哨兵，或一组 Send。三种都会用到：
    END 用于停止被人工中止的运行；'planner' 用于被拒时回环；扇出 Send
    用于批准/编辑的情况。所有可能的目标都在该边的 path map 中声明。"""
    if state.get("_aborted"):
        return END
    if state.get("_plan_rejected"):
        return "planner"
    return route_to_researchers(state)

def route_after_critic(state) -> str:
    """当研究发现已足够、达到迭代上限、或预算护栏触发时——以先到者为准——
    停止研究循环。任一情况都会流向 synthesizer，确保基于已有材料仍能
    产出报告。"""
    budget_spent = state.get("llm_calls", 0) >= settings.max_llm_calls_per_run
    if state.get("_sufficient") or state["iterations"] >= settings.max_research_iterations or budget_spent:
        return "synthesizer"
    return "planner"

def route_after_output_guard(state) -> str:
    """输出护栏：未通过的报告退回修复，否则交付。"""
    return "synthesizer" if state.get("_needs_repair") else END

def build_graph(checkpointer=None):
    """编译研究图。

    ``checkpointer`` 传入一个 saver（如 ``research_agent.persistence.sqlite_saver``
    的返回值），图会在每个超级步之后持久化状态，使运行可持久化、可恢复，
    并能暂停等待人工。
    传 ``None``（默认）时，图的行为与之前完全一致——内存态、临时态——
    评估/CI 不受影响。注意：人在回路审批门只有在存在检查点器时才会真正
    暂停，因为 ``interrupt()`` 需要持久化状态才能恢复。"""
    g = StateGraph(ResearchState)
    g.add_node("input_guard", input_guard_node, retry_policy=RETRY_POLICY)
    g.add_node("planner", planner_node, retry_policy=RETRY_POLICY)
    g.add_node("approval", approval_node)   # 人在回路审批门；无 LLM 调用
    g.add_node("researcher", researcher_node, retry_policy=RETRY_POLICY)   # 退避处理扇出时的 429
    g.add_node("critic", critic_node, retry_policy=RETRY_POLICY)
    g.add_node("synthesizer", synthesizer_node, retry_policy=RETRY_POLICY)
    g.add_node("output_guard", output_guard_node)   # 无 LLM 调用 → 无需重试

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input_guard, ["planner", END])
    g.add_edge("planner", "approval")
    g.add_conditional_edges("approval", route_after_approval, ["planner", "researcher", END])
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["planner", "synthesizer"])
    g.add_edge("synthesizer", "output_guard")
    g.add_conditional_edges("output_guard", route_after_output_guard, ["synthesizer", END])
    return g.compile(checkpointer=checkpointer)


def _print_result(result):
    print(result["report"])
    if result.get("guardrail_flags"):
        print("\n⚠ 护栏标记：", result["guardrail_flags"])
    print(f"\n本次运行 LLM 调用次数：{result.get('llm_calls')}")


def _load_query() -> str:
    """解析 CLI 演示的研究问题。

    优先级：命令行参数 > src/research-topic.md（位于本包旁边，因此任何
    工作目录均可运行）。文件中以 '#' 开头的行是注释；其余文本为研究问题。
    文件缺失或只有注释时大声退出——静默地研究错误的主题比停下来更糟。
    """
    argv_query = " ".join(sys.argv[1:]).strip()
    if argv_query:
        return argv_query

    topic_file = Path(__file__).resolve().parents[1] / "research-topic.md"
    if not topic_file.exists():
        sys.exit(f"✗ 未找到 {topic_file} — 请在该文件中写下研究问题")
    lines = [
        line.strip()
        for line in topic_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    query = " ".join(lines).strip()
    if not query:
        sys.exit(f"✗ {topic_file} 里没有研究问题（只有注释）— 请在文件正文中写下要研究的内容")
    return query


def _ask_human(payload):
    """渲染暂停的计划，并阻塞在 stdin 上等待真实的人工决策。

    返回审批节点 interrupt() 所期望的字典：
    {"action": "approve"} | {"action": "edit", "subtasks": [...]}
                          | {"action": "reject", "feedback": "..."}
                          | {"action": "abort", "reason": "..."}
    """
    print("\n⏸  图已暂停，等待人工审批。")
    print(f"研究问题：{payload.get('query')}")
    print("拟定子任务：")
    for i, task in enumerate(payload["proposed_subtasks"], 1):
        print(f"  {i}. {task}")

    # input() 会阻塞直到你输入内容并按回车——这就是等待点。
    choice = input("\n决策 [approve=通过 / edit=修改 / reject=打回 / abort=中止]（回车默认通过）: ").strip().lower()

    if choice == "edit":
        raw = input("新子任务（用逗号分隔）：")
        subtasks = [s.strip() for s in raw.split(",") if s.strip()]
        return {"action": "edit", "subtasks": subtasks}
    if choice == "reject":
        feedback = input("给规划节点的修改反馈：").strip()
        return {"action": "reject", "feedback": feedback}
    if choice == "abort":
        reason = input("中止原因（可留空）：").strip()
        return {"action": "abort", "reason": reason}
    return {"action": "approve"}


if __name__ == "__main__":
    # 开启 HITL（默认关闭），并把图包进 SQLite 检查点器，
    # 使运行在暂停期间得以保存。`thread_id` 标识这一次运行；
    # 复用它正是运行可恢复的关键。
    import uuid

    from langgraph.types import Command

    from research_agent.persistence import sqlite_saver

    settings.hitl_approval_enabled = True   # 仅演示用；正常情况下通过 env/config 设置

    query = _load_query()
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "run_name": "deep-research",
        "metadata": {
            "model_planner": settings.model_planner,
            "model_researcher": settings.model_researcher,
            "model_synthesizer": settings.model_synthesizer,
            "model_critic": settings.model_critic,
        },
    }

    progress(f"▶ 开始深度研究（运行 ID：{thread_id}）")
    progress(f"▶ 研究问题：{query}")
    progress(
        f"▶ 模型配置：planner={settings.model_planner} | researcher={settings.model_researcher} "
        f"| critic={settings.model_critic} | synthesizer={settings.model_synthesizer}"
    )

    with sqlite_saver() as saver:
        app = build_graph(checkpointer=saver)

        # 启动运行。它依次执行 input_guard → planner → approval，然后
        # approval 节点调用 interrupt()，图在此 PAUSE（暂停）并把控制权交回这里。
        result = app.invoke(cast("ResearchState", {"query": query}), config=config)

        # 暂停会在结果中表现为 __interrupt__ 载荷。只要图处于暂停状态就循环：
        # 询问人工，带着其决策恢复运行，如此往复。
        # （"reject" 会重新规划并再次暂停，这正是需要循环的原因。）
        while result.get("__interrupt__"):
            decision = _ask_human(result["__interrupt__"][0].value)
            print(f"\n▶  已选择操作：{decision['action']}，继续运行\n")
            result = app.invoke(Command(resume=decision), config=config)

        progress("▶ 研究完成，输出最终报告：")
        _print_result(result)