import time
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from research_agent.guardrails import LLMCallCounter
from research_agent.models import get_model
from research_agent.progress import progress
from research_agent.tools.search import web_search


@lru_cache(maxsize=1)
def _agent():
    """惰性创建研究员智能体：模块导入不再要求 OPENAI_API_KEY。"""
    return create_agent(get_model("researcher"), tools=[web_search])


def researcher_node(state):
    task = state["subtasks"][0]
    progress(f"🔬 研究员：开始研究「{task}」")
    # 统计智能体内部的模型调用次数（工具循环 → 次数不可预知），
    # 让预算护栏看到本分支的真实开销。reducer 会在扇出分支间求和。
    counter = LLMCallCounter()
    started = time.perf_counter()
    result = _agent().invoke(
        {"messages": [HumanMessage(content=f"请深入研究以下任务，并用中文汇总研究发现：{task}")]},
        config={"callbacks": [counter]},
    )
    elapsed = time.perf_counter() - started
    finding = result["messages"][-1].content
    progress(
        f"✔ 研究员：完成「{task}」（耗时 {elapsed:.1f}s，"
        f"{counter.count} 次模型调用，产出 {len(finding)} 字）"
    )
    return {"findings": [finding], "llm_calls": counter.count}