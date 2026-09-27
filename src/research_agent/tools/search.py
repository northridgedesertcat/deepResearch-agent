from functools import lru_cache

from langchain.tools import tool
from langchain_tavily import TavilySearch

from research_agent.progress import progress


@lru_cache(maxsize=1)
def _search():
    """惰性创建搜索工具：模块导入不再要求 TAVILY_API_KEY，
    单元测试（桩图、无网络）也能正常收集。"""
    return TavilySearch(max_results=4)


@tool("web_search", description="联网搜索某个问题的最新信息。")
def web_search(query: str) -> str:
    progress(f"🔍 联网搜索：{query}")
    return _search().invoke({"query": query})