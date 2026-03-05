try:
    from langchain_core.tools import tool
except ImportError:
    def tool(f):
        return f  # no-op when langchain not installed

from playwright.async_api import Page


async def _navigate(url: str, page: Page) -> str:
    """导航到指定的URL（内部实现）。"""
    await page.goto(url)
    return f"已导航至 {url}"


@tool
async def navigate(url: str, page: Page) -> str:
    """导航到指定的URL。"""
    return await _navigate(url, page)

@tool
async def click(element_index: int, page: Page) -> str:
    """点击页面上由索引指定的元素。"""
    # 这里需要结合标注逻辑，找到对应索引的元素并点击 [citation:7]
    # ...
    return f"已点击元素 {element_index}"

@tool
async def type_text(input_text: str, page: Page) -> str:
    """在当前的输入框中输入文本（通常先点击再输入）。"""
    # ...
    return f"已输入文本: {input_text}"

@tool
async def scroll(direction: str, page: Page) -> str:
    """向下或向上滚动页面。"""
    # ...
    return f"已向{direction}滚动"


URL = "https://monitoring.infra.sz.shopee.io/grafana/d/zk-streamline/zk-streamline?from=now-5m&orgId=74&to=now&var-DS_PROMETHEUS=middleware_consul&var-cluster=zk-mp-search-recommendation-ads-engineering-and-architecture-live-6m4m9twz-cc-backup&var-env=live"

async def main():
    import asyncio
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)  # 有界面模式，会弹出浏览器窗口
        page = await browser.new_page()
        result = await _navigate(URL, page)
        print(result)
        # 保持窗口打开几秒，便于查看页面（可改成 input("按回车关闭...") 手动关闭）
        await asyncio.sleep(30)
        await browser.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())