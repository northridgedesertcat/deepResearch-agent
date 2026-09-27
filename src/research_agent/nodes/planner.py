from pydantic import BaseModel, Field
from research_agent.models import invoke_structured
from research_agent.progress import progress

class Plan(BaseModel):
    subtasks: list[str] = Field(description="3-5 个聚焦、互不重叠的研究子问题")

def planner_node(state):
    feedback = state.get("plan_feedback")   # ← 人工拒绝了上一版计划时设置（HITL）
    gaps = state.get("gaps")                # ← 首轮为空，critic 回环之后才有值
    if feedback:
        progress("📋 规划器：根据人工反馈重新拟定子任务...")
        prompt = (
            f"原始研究问题：{state['query']}\n\n"
            f"人工审核者拒绝了上一版计划，反馈如下：\n{feedback}\n\n"
            "请根据反馈重新拟定研究子任务。"
        )
    elif gaps:
        progress("📋 规划器：针对上一轮研究遗留的缺口拟定补充子任务...")
        prompt = (
            f"原始研究问题：{state['query']}\n\n"
            f"第一轮研究遗留以下缺口：\n{gaps}\n\n"
            "请只针对这些缺口拟定聚焦的研究子任务。"
        )
    else:
        progress("📋 规划器：正在拆解研究问题...")
        prompt = f"请将以下研究问题拆解为研究子任务：\n{state['query']}"

    plan = invoke_structured("planner", Plan, prompt)
    assert isinstance(plan, Plan)
    for i, task in enumerate(plan.subtasks, 1):
        progress(f"   ├─ 子任务 {i}/{len(plan.subtasks)}：{task}")
    # 清空 plan_feedback，避免泄露到之后 critic 回环的重新规划中。
    return {"subtasks": plan.subtasks, "llm_calls": 1, "plan_feedback": ""}   # 预算记账