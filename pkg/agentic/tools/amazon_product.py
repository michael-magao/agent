"""
亚马逊商品分析工具 - 基于 Amazon Product Advertising API 5.0
抓取商品信息并分析，输出可开发投入市场的产品建议。
"""
from __future__ import annotations

import json
import os
from typing import Optional

# 可选依赖：amazon-paapi5 (pip install amazon-paapi5)，未安装时使用 mock 模式
try:
    from amazon.paapi import AmazonAPI
    _PAAPI_AVAILABLE = True
except ImportError:
    AmazonAPI = None
    _PAAPI_AVAILABLE = False


def _get_paapi_client() -> Optional["AmazonAPI"]:
    """获取 PA-API 客户端，缺省配置时返回 None"""
    key = os.getenv("AMAZON_PAAPI_ACCESS_KEY") or os.getenv("AMAZON_ACCESS_KEY")
    secret = os.getenv("AMAZON_PAAPI_SECRET_KEY") or os.getenv("AMAZON_SECRET_KEY")
    tag = os.getenv("AMAZON_PAAPI_PARTNER_TAG") or os.getenv("AMAZON_PARTNER_TAG")
    country = os.getenv("AMAZON_PAAPI_COUNTRY", "US")
    if not (key and secret and tag):
        return None
    if not _PAAPI_AVAILABLE:
        return None
    return AmazonAPI(access_key=key, secret_key=secret, partner_tag=tag, country=country)


def _mock_search_results(keyword: str) -> dict:
    """Mock 数据，用于未配置 PA-API 时的演示"""
    return {
        "keyword": keyword,
        "total_results": 5,
        "items": [
            {
                "title": f"Example Product A - {keyword}",
                "asin": "B0MOCK01",
                "price": 29.99,
                "rating": 4.5,
                "reviews_count": 1234,
                "brand": "BrandA",
            },
            {
                "title": f"Example Product B - {keyword}",
                "asin": "B0MOCK02",
                "price": 19.99,
                "rating": 4.2,
                "reviews_count": 890,
                "brand": "BrandB",
            },
        ],
    }


def _normalize_paapi_response(data: any) -> dict:
    """将 PA-API 返回结构转为统一格式。支持 amazon-paapi5 的 data 列表或原生 items"""
    items = []
    raw_items = []
    if isinstance(data, list):
        raw_items = data[:10]
    elif isinstance(data, dict):
        raw_items = data.get("items", data.get("data", []))[:10]
    else:
        raw_items = (getattr(data, "items", None) or getattr(data, "data", None) or [])[:10]

    def _get(obj, *path, default=None):
        for k in path:
            if obj is None:
                return default
            if isinstance(k, int):
                obj = obj[k] if isinstance(obj, (list, tuple)) and 0 <= k < len(obj) else None
            elif isinstance(obj, dict):
                obj = obj.get(k)
            else:
                obj = getattr(obj, k, None)
        return obj if obj is not None else default

    for it in raw_items:
        # title
        title = _get(it, "item_info", "title", "display_value") or _get(it, "title") or getattr(it, "title", "") or ""
        if callable(title):
            title = ""
        title = str(title)[:200] if title else ""

        # brand
        brand = _get(it, "item_info", "by_line_info", "brand", "display_value") or _get(it, "brand") or getattr(it, "brand", "") or ""
        brand = str(brand) if brand else ""

        # price - amazon-paapi5 用 prices.price
        price = 0.0
        p = _get(it, "offers", "listings", 0, "price", "amount") or _get(it, "prices", "price") or getattr(getattr(getattr(it, "prices", None), "price", None), "amount", None)
        if p is not None:
            try:
                price = float(p)
            except (TypeError, ValueError):
                pass

        # rating
        rating = 0.0
        r = _get(it, "customer_reviews", "star_rating", "value") or _get(it, "ratings", "value")
        if r is not None:
            try:
                rating = float(r)
            except (TypeError, ValueError):
                pass

        # reviews_count
        reviews_count = 0
        rc = _get(it, "customer_reviews", "reviews_count") or getattr(getattr(it, "customer_reviews", None), "reviews_count", None)
        if rc is not None:
            try:
                reviews_count = int(rc)
            except (TypeError, ValueError):
                pass

        asin = _get(it, "asin") or getattr(it, "asin", "") or ""
        asin = str(asin) if asin else ""
        items.append({
            "asin": asin,
            "title": title,
            "brand": brand,
            "price": price,
            "rating": rating,
            "reviews_count": reviews_count,
        })

    return {"keyword": getattr(data, "keyword", "") or "", "total_results": len(items), "items": items}


def amazon_search_products(keyword: str, marketplace: str = "US") -> str:
    """
    搜索亚马逊商品。基于 PA-API 5.0 SearchItems。
    需设置环境变量：AMAZON_PAAPI_ACCESS_KEY, AMAZON_PAAPI_SECRET_KEY, AMAZON_PAAPI_PARTNER_TAG
    """
    client = _get_paapi_client()
    if client:
        try:
            resp = client.search_items(keywords=keyword)
            # amazon-paapi5 返回 {"data": [...]} 或类似结构
            data = resp.get("data", resp) if isinstance(resp, dict) else resp
            normalized = _normalize_paapi_response(data)
        except Exception as e:
            return json.dumps({"error": str(e), "keyword": keyword}, ensure_ascii=False, indent=2)
    else:
        normalized = _mock_search_results(keyword)

    return json.dumps(normalized, ensure_ascii=False, indent=2)


def amazon_analyze_for_product_dev(keyword: str, marketplace: str = "US") -> str:
    """
    抓取亚马逊商品信息并分析，输出可开发投入市场的产品建议。
    输出结构：市场概览、竞品分析、用户痛点、产品开发建议、投入产出评估。
    """
    data_json = amazon_search_products(keyword, marketplace)
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        return data_json

    if "error" in data:
        return data_json

    # 构造供 LLM 分析的摘要（后续由 Agent 调用 LLM 做深度分析）
    summary_lines = [
        f"## 关键词: {keyword}",
        f"## 检索商品数: {data.get('total_results', 0)}",
        "",
        "### 竞品样本",
    ]
    for i, it in enumerate(data.get("items", [])[:10], 1):
        summary_lines.append(
            f"{i}. [{it.get('title','')}] 品牌:{it.get('brand','')} "
            f"价格:${it.get('price',0):.2f} 评分:{it.get('rating',0)} 评论数:{it.get('reviews_count',0)}"
        )

    raw_summary = "\n".join(summary_lines)
    full_json = json.dumps(data, ensure_ascii=False, indent=2)

    # 返回结构化数据，供 Agent 的 LLM 进一步分析生成产品开发建议
    return f"""{raw_summary}

---
完整原始数据(JSON):
{full_json}

---
请基于以上数据，从以下维度生成《产品开发投入建议报告》：
1. 市场概览（类目规模、竞争密度）
2. 竞品分析（价格带、评分、销量信号）
3. 用户痛点（从评价/评论可推断的需求）
4. 产品开发建议（差异化方向、价格带、功能建议）
5. 投入产出评估（竞争难度、预估机会、风险提示）
"""
