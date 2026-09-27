"""健壮性 + 输出检查（不依赖图结构）。

与护栏*节点*（``nodes/guardrails.py``）分开存放，使这些纯函数辅助工具
保持易于测试：

- ``RETRY_POLICY``      — LangGraph 重试策略，针对瞬态失败（*包括 429 限流*）
                          进行退避重试。
- ``LLMCallCounter``    — 统计模型调用次数的回调，让 researcher 子智能体
                          能向预算护栏上报真实开销。
- ``verify_citations``  — 硬性引用校验门：找出哪些被引用的 URL 没有已采集
                          来源支撑。
- ``scan_content``      — 内容检查：系统提示词泄露与个人敏感信息（PII）。
"""

from __future__ import annotations

import re

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import RetryPolicy

from research_agent.config import settings


# ── 健壮性：重试策略 ─────────────────────────────────────────────────────────
def _is_transient(exc: Exception) -> bool:
    """对限流（429）、服务端错误（5xx）和连接中断进行重试。

    各厂商 SDK 没有共同的基类，因此采用结构化匹配：
    任何携带 429 / 5xx 状态码的异常，加上任何名字里带"限流"或"超时"
    的错误。确定性 bug（ValueError、TypeError 等）刻意*不*重试——
    重试它们只是浪费钱。
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "timeout" in name


# 一份策略，图中所有 LLM 节点复用。`initial_interval` 每次尝试后按
# `backoff_factor` 翻倍 → 1s、2s、4s、8s，正是预期的退避节奏。
RETRY_POLICY = RetryPolicy(
    max_attempts=settings.retry_max_attempts,
    initial_interval=settings.retry_initial_interval,
    backoff_factor=settings.retry_backoff_factor,
    retry_on=_is_transient,
)


# ── 预算：统计真实 LLM 调用次数 ─────────────────────────────────────────────
class LLMCallCounter(BaseCallbackHandler):
    """统计单次 ``invoke`` 内的模型调用次数。

    普通 LLM 触发 ``on_llm_start``；聊天模型触发 ``on_chat_model_start``——
    一次调用只会触发其中之一，因此两者都计数是安全的，绝不会重复计数。
    供 researcher 节点使用：其内部工具循环的调用次数不可预知，
    需要向预算护栏上报真实开销。
    """

    def __init__(self) -> None:
        self.count = 0

    def on_llm_start(self, *args, **kwargs) -> None:
        self.count += 1

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.count += 1


# ── 输出护栏：引用校验 ───────────────────────────────────────────────────────
def _findings_text(findings) -> str:
    """把累积的研究发现（字符串或 Finding 对象）展平成一段文本。"""
    parts = []
    for f in findings or []:
        parts.append(f if isinstance(f, str) else str(getattr(f, "model_dump", lambda: f)()))
    return "\n\n".join(parts)


def verify_citations(findings, report) -> tuple[float, list[str]]:
    """硬性引用校验门。

    返回 ``(coverage, unsupported)``：``coverage`` 是报告中被引用 URL
    确实出现在已采集研究结果中的比例；``unsupported`` 列出没有出现的
    URL——即模型可能凭空编造的引用。没有任何引用的报告得分为 0.0：
    无依据的报告应当无法通过此门。
    """
    citations = report.get("citations", []) if isinstance(report, dict) else []
    if not citations:
        return 0.0, []
    haystack = _findings_text(findings)
    unsupported = [url for url in citations if not url or url not in haystack]
    coverage = (len(citations) - len(unsupported)) / len(citations)
    return coverage, unsupported


# ── 输出护栏：内容检查 ───────────────────────────────────────────────────────
# 出现这些标记意味着我们自己的指令泄露到了面向用户的文本中。
_PROMPT_LEAK_MARKERS = (
    "you are a strict evaluator",
    "system prompt",
    "every claim must reference",
    "break this into research subtasks",
    "as an ai language model",
)

# 保守的 PII 匹配模式。邮箱和长数字串（电话/银行卡号）是网络研究报告
# 中现实存在的泄露风险；刻意保持简单。
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LONG_DIGITS_RE = re.compile(r"\b(?:\d[ -]?){9,}\b")


def scan_content(text: str) -> list[str]:
    """返回报告文本中发现的内容违规列表（空列表 = 干净）。"""
    flags: list[str] = []
    low = text.lower()
    for marker in _PROMPT_LEAK_MARKERS:
        if marker in low:
            flags.append(f"possible leaked system prompt: '{marker}'")
    if _EMAIL_RE.search(text):
        flags.append("possible PII: email address in report")
    if _LONG_DIGITS_RE.search(text):
        flags.append("possible PII: phone/card-like number in report")
    return flags
