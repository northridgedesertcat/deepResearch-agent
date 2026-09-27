"""
三个评分器，各自返回 [0, 1] 区间的浮点数：

- ``faithfulness``      — LLM-as-judge：报告中的论断是否有研究者实际采集的
                          研究发现支撑？
- ``coverage``          — LLM-as-judge：报告是否覆盖了问题的每个部分
                          （对照数据集的 ``key_points`` 检查）？
- ``citation_accuracy`` — 程序化检查（无需评审模型、零成本）：报告引用的 URL
                          是否确实出现在已采集的研究发现中？这是直接、
                          确定性的测量。

两个 LLM 评审与系统其余部分一样，走同一个与提供商无关的工厂
（``invoke_structured("judge", ...)``），通过 ``model_judge`` 指向廉价的模型
规格，并复用 ``invoke_structured`` 的阶梯式结构化输出重试策略。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_agent.models import invoke_structured


class Score(BaseModel):
    """评审的判定：0..1 的分数，加一行理由（用于追踪日志）。"""

    score: float = Field(ge=0.0, le=1.0, description="0.0 = 不符合评分标准，1.0 = 完全满足")
    reasoning: str = Field(description="为该分数给出的一句话理由")


def _findings_text(findings) -> str:
    """把累积的研究发现（字符串或 Finding 类对象）展平成一段文本。"""
    parts = []
    for f in findings or []:
        parts.append(f if isinstance(f, str) else getattr(f, "model_dump", lambda: f)().__str__())
    return "\n\n".join(parts)


def _report_text(report) -> str:
    """把报告字典（summary + sections）渲染成纯文本，供评审阅读。"""
    if isinstance(report, dict):
        summary = report.get("summary", "")
        sections = [
            f"{s.get('title', '')}\n{s.get('content', '')}" if isinstance(s, dict) else str(s)
            for s in (report.get("sections", []) or [])
        ]
        joined = "\n".join(sections)
        return f"{summary}\n\n{joined}".strip()
    return str(report)


def faithfulness(findings, report) -> Score:
    """报告中的论断是否有已采集的研究发现支撑，还是凭空编造？"""
    verdict = invoke_structured(
        "judge",
        Score,
        "You are a strict evaluator scoring GROUNDEDNESS.\n"
        "Score from 0 to 1 how well EVERY claim in the report is supported by the "
        "findings below. Penalise claims that are not backed by the findings.\n\n"
        f"FINDINGS:\n{_findings_text(findings)}\n\n"
        f"REPORT:\n{_report_text(report)}",
    )
    assert isinstance(verdict, Score)
    return verdict


def coverage(query: str, key_points: list[str], report) -> Score:
    """报告是否覆盖了整个问题以及预期要点？"""
    points = "\n".join(f"- {p}" for p in key_points)
    verdict = invoke_structured(
        "judge",
        Score,
        "You are a strict evaluator scoring COVERAGE.\n"
        "Score from 0 to 1 the fraction of the expected key points that the report "
        "meaningfully addresses for the given query.\n\n"
        f"QUERY:\n{query}\n\n"
        f"EXPECTED KEY POINTS:\n{points}\n\n"
        f"REPORT:\n{_report_text(report)}",
    )
    assert isinstance(verdict, Score)
    return verdict


def citation_accuracy(findings, report) -> float:
    """被引用的 URL 中确实出现在已采集研究发现里的比例。

    确定性且免费——不调用 LLM。没有任何引用的报告得分为 0，
    这正是我们想要的行为：无依据的报告应当无法通过此门。
    """
    citations = report.get("citations", []) if isinstance(report, dict) else []
    if not citations:
        return 0.0
    haystack = _findings_text(findings)
    present = sum(1 for url in citations if url and url in haystack)
    return present / len(citations)
