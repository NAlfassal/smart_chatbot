import asyncio
from typing import Optional, Dict
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

class SDISearchTool:
    """
    Tool to search Saudi Drugs Information System (SDI).
    Implements browser automation to fetch drug details by Trade Name.
    """

    def __init__(self):
        self.sdi_url = "https://sdi.sfda.gov.sa/Home/AdvancedSearch"
        
        # UI selectors based on element IDs
        self.selectors = {
            "trade_name_input": "#tradeName",
            "search_button": "#search",
            "table_container": ".table-responsive",
            "table_rows": ".table-responsive table tbody tr",
            "table_headers": ".table-responsive table thead th",
            # Text indicators for zero results
            "no_results_indicators": [
                "من إجمالي 0",
                "of 0 entries",
                "No matching records",
                "لا توجد سجلات مطابقة"
            ]
        }

    async def _extract_table_data(self, page) -> Optional[str]:
        """Helper to parse the results table."""
        try:
            # Wait for table visibility
            await page.wait_for_selector(self.selectors["table_container"], state="visible", timeout=5000)
            
            # Get headers and first row cells
            headers = await page.eval_on_selector_all(
                self.selectors["table_headers"], 
                "elements => elements.map(e => e.innerText.trim())"
            )
            
            cells = await page.eval_on_selector_all(
                f"{self.selectors['table_rows']}:first-child td", 
                "elements => elements.map(e => e.innerText.trim())"
            )

            if not cells:
                return None

            # Format the output
            result_parts = []
            for h, c in zip(headers, cells):
                if h and c and "View" not in h and "مشاهدة" not in h:
                    result_parts.append(f"🔹 **{h}**: {c}")
            
            return "\n".join(result_parts)
            
        except Exception:
            return None

    async def search_drug(self, trade_name: str) -> Dict[str, str]:
        """
        Main function to perform the search.
        Returns a dictionary with status and message.
        """
        async with async_playwright() as p:
            # Configure browser to mimic a real user (Stealth Mode)
            # This is necessary to bypass government WAF security
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    channel="chrome", 
                    args=["--disable-blink-features=AutomationControlled"]
                )
            except Exception:
                browser = await p.chromium.launch(headless=True)

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            try:
                # 1. Open the search page
                await page.goto(self.sdi_url, wait_until="domcontentloaded", timeout=60000)

                # 2. Input the trade name
                input_field = await page.wait_for_selector(self.selectors["trade_name_input"], state="visible")
                await input_field.fill("") 
                await input_field.type(trade_name, delay=50)

                # 3. Execute search
                await page.click(self.selectors["search_button"])

                # 4. Check results
                try:
                    await page.wait_for_selector(self.selectors["table_container"], timeout=10000)
                    
                    # Check for 'No Results' text in the page
                    page_content = await page.content()
                    for indicator in self.selectors["no_results_indicators"]:
                        if indicator in page_content:
                            await browser.close()
                            return {
                                "status": "not_found",
                                "message": f"الدواء ({trade_name}) غير مسجل."
                            }

                    # Extract data if results exist
                    data = await self._extract_table_data(page)
                    await browser.close()
                    
                    if data:
                        return {
                            "status": "success",
                            "message": f"الدواء مسجل\n\n{data}"
                        }
                    else:
                        return {
                            "status": "not_found",
                            "message": f"لاتوجد معلومات عن الدواء لدى SDI-SFDA , الرجاء التأكد من صحة الإسم المدخل ({trade_name})."
                        }

                except PlaywrightTimeoutError:
                    await browser.close()
                    return {
                        "status": "error", 
                        "message": "Search request timed out."
                    }

            except Exception as e:
                await browser.close()
                return {
                    "status": "error", 
                    "message": f"Technical Error: {str(e)}"
                }

# Test execution
if __name__ == "__main__":
    async def main():
        tool = SDISearchTool()
        result = await tool.search_drug("VarNell")
        print(result["message"])

    asyncio.run(main())