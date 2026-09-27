from typing import Annotated, TypedDict
from operator import add

from pydantic import BaseModel

class Finding(BaseModel):
    claim: str
    source_url: str
    source_title: str

class ResearchState(TypedDict, total=False):
    query: str                                # 在输入时设置
    subtasks: list[str]                       # 由 planner 写入
    findings: Annotated[list[Finding], add]   # ← reducer：累积追加
    report: dict                              # 由 synthesizer 写入（summary/sections/citations）
    iterations: int                           # 每轮 critic 检查后递增
    gaps: list[str]                           # critic 留给下一轮的备注

    # 护栏
    rejected: bool                            # 输入护栏的判定结果
    rejection_reason: str                     # 拒绝查询时返回给用户的消息
    llm_calls: Annotated[int, add]            # 预算记账（扇出时各分支求和）
    repairs: int                              # 输出护栏目前已尝试的修复次数
    repair_notes: str                         # 输出护栏 → synthesizer 的修复反馈
    guardrail_flags: list[str]                # 报告上暴露的引用/内容问题

    # 人在回路
    plan_feedback: str                        # 计划被拒时人工的备注 → planner 重新起草
    _plan_rejected: bool                      # 审批门设置的临时路由标志
    _aborted: bool                            # 人工在审批门中止 → 直接路由到 END