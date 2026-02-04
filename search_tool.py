import asyncio
from typing import Optional, Dict
from playwright.async_api import async_playwright

class SDISearchAgent:
    """
    Search Agent for Saudi Drugs Information System (SDI).
    Targeting elements based on verified HTML IDs.
    """

    def __init__(self):
        self.sdi_url = "https://sdi.sfda.gov.sa/Home/AdvancedSearch"
        
        self.selectors = {
            "trade_name_input": "#tradeName",   
            "search_button": "#search",          
            "table_container": ".table-responsive", 
            "table_rows": ".table-responsive table tbody tr",
            "table_headers": ".table-responsive table thead th",
        }

    async def _extract_table_data(self, page) -> Optional[str]:
        """Extracts data mapping headers to the first row cells."""
        try:
            # Wait for the table container to be visible
            await page.wait_for_selector(self.selectors["table_container"], state="visible", timeout=5000)
            
            # Get Headers (رقم التسجيل, الاسم التجاري, ...)
            headers = await page.eval_on_selector_all(
                self.selectors["table_headers"], 
                "elements => elements.map(e => e.innerText.trim())"
            )
            
            # Get First Row Data
            cells = await page.eval_on_selector_all(
                f"{self.selectors['table_rows']}:first-child td", 
                "elements => elements.map(e => e.innerText.trim())"
            )

            if not cells: return None

            result_parts = []
            for h, c in zip(headers, cells):
                # Filter out empty columns or the 'View/مشاهدة' button column
                if h and c and "View" not in h and "مشاهدة" not in c:
                    result_parts.append(f"🔹 **{h}**: {c}")
            
            return "\n".join(result_parts)
        except Exception:
            return None

    async def search_by_trade_name(self, trade_name: str) -> Dict[str, str]:
        """
        Executes the search workflow using specific IDs.
        """
        async with async_playwright() as p:
            # Using Chrome channel to bypass WAF (Security)
            try:
                browser = await p.chromium.launch(
                    headless=True, 
                    channel="chrome", 
                    args=["--disable-blink-features=AutomationControlled"]
                )
            except:
                # Fallback if Chrome is not installed
                browser = await p.chromium.launch(headless=True)

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            try:
                # 1. Open URL
                print("🌐 Connecting to SDI System...")
                await page.goto(self.sdi_url, wait_until="domcontentloaded", timeout=60000)

                # 2. Fill Trade Name
                print(f"✍️ Entering Trade Name: {trade_name}")
                input_field = await page.wait_for_selector(self.selectors["trade_name_input"], state="visible")
                await input_field.fill("")
                await input_field.type(trade_name, delay=50) # Typing like a human

                # 3. Click Search Button (
                print("🔍 Clicking Search Button...")
                await page.click(self.selectors["search_button"])

                # 4. Wait for Results
                try:
                    # Wait for either the Table or "no matching records" message
                    found_element = await page.wait_for_selector(
                        f"{self.selectors['table_container']}, .dataTables_empty", 
                        timeout=15000
                    )
                    
                    # Check text content for "No results"
                    content = await page.content()
                    if "No matching records found" in content or "لا توجد سجلات مطابقة" in content:
                         await browser.close()
                         return {
                             "status": "not_found",
                             "message": f"❌ لم يتم العثور على دواء بالاسم: {trade_name}"
                         }

                    # Extract Data
                    data = await self._extract_table_data(page)
                    await browser.close()
                    
                    if data:
                        return {
                            "status": "success",
                            "message": f"✅ تم العثور على الدواء:\n\n{data}"
                        }
                    else:
                        return {
                            "status": "not_found",
                            "message": "⚠️ ظهر الجدول ولكن لم يتم استخراج البيانات بشكل صحيح."
                        }

                except Exception:
                    await browser.close()
                    return {"status": "error", "message": "⚠️ انتهى الوقت ولم تظهر النتائج."}

            except Exception as e:
                await browser.close()
                return {"status": "error", "message": f"⚠️ خطأ فني: {str(e)}"}

# ==========================================
# Testing
# ==========================================
if __name__ == "__main__":
    async def main():
        agent = SDISearchAgent()
        
        # Test with "Posaviv" as shown in your screenshot
        result = await agent.search_by_trade_name("varnell")
        
        print("\n" + "="*40)
        print(result["message"])
        print("="*40 + "\n")

    asyncio.run(main())