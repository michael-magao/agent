"""
Web 搜索工具：基于互联网检索信息，供 Agent 在需要最新或外部信息时调用，提高执行准确性。
支持 DDGS（免费）与可选的 Tavily API（需配置 TAVILY_API_KEY）。
"""
from __future__ import annotations

import os
from typing import Optional

# 默认返回条数，兼顾信息量与 token 消耗
DEFAULT_MAX_RESULTS = 8


def _search_with_ddgs(query: str, max_results: int) -> list[dict]:
    """使用 DDGS（duckduckgo-search / ddgs）进行网页搜索，无需 API Key。"""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "请安装搜索依赖: pip install ddgs  或  pip install duckduckgo-search"
            ) from None

    with DDGS() as ddgs:
        # text() 返回 list[dict]，键一般为 title, href, body
        raw = ddgs.text(query, max_results=max_results)
    return list(raw) if raw else []


def _search_with_tavily(query: str, max_results: int) -> list[dict]:
    """使用 Tavily API 进行搜索（需设置 TAVILY_API_KEY）。"""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except ImportError:
        return []

    tool = TavilySearchResults(
        max_results=max_results,
        search_depth="basic",
        include_answer=True,
        include_raw_results=False,
    )
    try:
        result = tool.invoke({"query": query})
    except Exception:
        return []
    # Tavily 返回格式: [{"url": ..., "content": ...}, ...]
    if not result:
        return []
    return [
        {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
        for r in (result if isinstance(result, list) else [result])
    ]


def _format_results(query: str, items: list[dict], backend: str) -> str:
    """将搜索结果格式化为 Agent 易读的文本。"""
    if not items:
        return f"未找到与「{query}」相关的搜索结果。"

    lines = [f"搜索查询: {query}（来源: {backend}）", ""]
    for i, item in enumerate(items, 1):
        title = item.get("title") or item.get("name") or ""
        url = item.get("href") or item.get("url") or ""
        body = item.get("body") or item.get("content") or item.get("snippet") or ""
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   链接: {url}")
        if body:
            lines.append(f"   摘要: {body.strip()[:500]}")
        lines.append("")
    return "\n".join(lines).strip()


def web_search(
    query: str,
    max_results: Optional[int] = None,
) -> str:
    """基于互联网的 Web 搜索工具。

    当 Agent 需要以下信息时应调用本工具：
    - 实时/最新信息（新闻、行情、文档更新等）
    - 非内部知识库的公开事实、定义、教程
    - 外部网站内容、产品信息、竞品与市场信息

    参数:
        query: 搜索关键词或问句，建议简洁明确。
        max_results: 返回结果数量，默认 8；可选 1–15。

    返回:
        格式化的搜索结果文本（标题、链接、摘要），便于后续推理与引用。
    """
    if not (query and str(query).strip()):
        return "搜索关键词不能为空。"

    n = max_results if max_results is not None else DEFAULT_MAX_RESULTS
    n = max(1, min(15, int(n)))

    # 若配置了 Tavily 则优先使用，否则使用 DDGS
    if os.environ.get("TAVILY_API_KEY"):
        items = _search_with_tavily(query, n)
        backend = "Tavily"
        if not items:
            items = _search_with_ddgs(query, n)
            backend = "DDGS"
    else:
        items = _search_with_ddgs(query, n)
        backend = "DDGS"

    return _format_results(query, items, backend)
