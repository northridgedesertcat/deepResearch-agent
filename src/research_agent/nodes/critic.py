from pydantic import BaseModel
from research_agent.config import settings
from research_agent.models import invoke_structured
from research_agent.progress import progress

class Assessment(BaseModel):
    sufficient: bool
    missing: list[str]   # 下一轮要研究的缺口；充分则为空

def critic_node(state):
    round_no = state.get("iterations", 0) + 1
    progress(f"🔍 评审员：正在评估第 {round_no}/{settings.max_research_iterations} 轮研究发现是否充分...")
    assessment = invoke_structured(
        "critic", Assessment,
        f"研究问题：{state['query']}\n\n研究发现：\n{state['findings']}\n\n"
        "这些材料是否充分且有可靠来源支撑？如不充分，请列出还缺什么。"
    )
    assert isinstance(assessment, Assessment)
    if assessment.sufficient:
        progress("✔ 评审员：研究材料已充分，进入报告撰写")
    else:
        gaps_text = "；".join(assessment.missing) or "（未具体列出）"
        progress(f"⟳ 评审员：材料尚不充分，缺口：{gaps_text}")
    return {"gaps": assessment.missing, "iterations": round_no,
            "llm_calls": 1,                          # 预算记账
            "_sufficient": assessment.sufficient}   # 供路由使用的临时标志
