from langchain.tools import tool
from langchain_tavily import TavilySearch

from research_agent.progress import progress

_search = TavilySearch(max_results=4)

@tool("web_search", description="联网搜索某个问题的最新信息。")
def web_search(query: str) -> str:
    progress(f"🔍 联网搜索：{query}")
    return _search.invoke({"query": query})