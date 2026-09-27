from pydantic import BaseModel

from research_agent.models import invoke_structured
from research_agent.progress import progress

class Section(BaseModel):
    title: str
    content: str

class CitedReport(BaseModel):
    summary: str
    sections: list[Section]
    citations: list[str]   # 实际使用的来源 URL

def synthesizer_node(state):
    prompt = (
        "请撰写一份中文研究报告，每个论断都必须引用下列来源的 URL。\n\n"
        f"研究发现：\n{state['findings']}"
    )
    # 输出护栏的修复轮次中，附上护栏的具体反馈。
    repair_notes = state.get("repair_notes")
    if repair_notes:
        progress("✍ 综合器：正在根据护栏反馈修复报告...")
        prompt += f"\n\n{repair_notes}"
    else:
        progress(f"✍ 综合器：正在撰写带引用的研究报告（基于 {len(state['findings'])} 份研究发现）...")

    report = invoke_structured("synthesizer", CitedReport, prompt)
    if not isinstance(report, CitedReport):
        raise ValueError(f"期望 CitedReport，实际得到 {type(report)}")
    return {"report": report.model_dump(), "llm_calls": 1}
