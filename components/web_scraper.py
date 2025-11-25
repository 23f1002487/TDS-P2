"""
Web Scraper Component
Handles web page fetching and HTML parsing with Playwright
"""
import asyncio
from typing import Optional, Dict, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from playwright.async_api import async_playwright, Browser
from bs4 import BeautifulSoup


class WebScraper:
    """
    Manages web scraping with Playwright for JS rendering and BeautifulSoup for parsing
    """
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
        logger.info("WebScraper initialized")
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("WebScraper cleanup complete")
    
    async def get_browser(self) -> Browser:
        """Get or create Playwright browser instance"""
        if not self.browser:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu'
                ]
            )
            logger.info("Playwright browser launched")
        return self.browser
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def fetch_page(self, url: str) -> str:
        """
        Fetch and render JavaScript-based page using Playwright
        Auto-retries with exponential backoff
        """
        logger.info(f"Fetching page: {url}")
        
        browser = await self.get_browser()
        page = await browser.new_page()
        
        try:
            # Navigate to page
            await page.goto(url, wait_until='networkidle', timeout=15000)
            
            # Wait for content to render
            await page.wait_for_load_state('domcontentloaded')
            await asyncio.sleep(2)  # Additional wait for JS execution
            
            # Get rendered HTML
            html_content = await page.content()
            
            logger.success(f"Page fetched successfully: {len(html_content)} bytes")
            return html_content
            
        except Exception as e:
            logger.error(f"Error fetching page: {e}")
            raise
        finally:
            await page.close()
    
    def parse_html(self, html_content: str) -> Dict[str, Any]:
        """Extract structured information from HTML"""
        logger.info("Parsing HTML content")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to find result div first
        result_div = soup.find(id='result')
        if result_div:
            text = result_div.get_text(separator='\n', strip=True)
            result_html = str(result_div)
            logger.info("Found content in #result div")
        else:
            text = soup.get_text(separator='\n', strip=True)
            result_html = html_content
            logger.info("Using full page text")
        
        # Extract links
        links = {}
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True)
            link_url = link['href']
            links[link_text] = link_url
            logger.debug(f"Found link: {link_text} -> {link_url}")
        
        # Extract all visible elements with IDs (might contain secret codes)
        elements_with_ids = {}
        for element in soup.find_all(id=True):
            elem_id = element.get('id')
            elem_text = element.get_text(strip=True)
            if elem_text:  # Only include non-empty elements
                elements_with_ids[elem_id] = elem_text
        
        # Also extract script tags (they might contain secrets in JS code)
        script_content = []
        for script in soup.find_all('script'):
            script_text = script.string
            if script_text and len(script_text.strip()) > 0:
                script_content.append(script_text.strip())
        
        logger.info(f"Extracted {len(elements_with_ids)} elements with IDs, {len(script_content)} script tags")
        
        # Log the elements for debugging
        if elements_with_ids:
            logger.debug("Elements with IDs found:")
            for elem_id, elem_text in elements_with_ids.items():
                logger.debug(f"  - #{elem_id}: {elem_text[:200]}")
        
        if script_content:
            logger.debug(f"Found {len(script_content)} script tags with content")
        
        return {
            'text': text,
            'links': links,
            'html': html_content,
            'result_html': result_html,
            'elements_with_ids': elements_with_ids,
            'script_content': script_content
        }
    
    def build_dynamic_context(self, quiz_info: Dict, task_info: Dict) -> tuple:
        """Build dynamic context and instructions based on page content"""
        page_text = quiz_info['text'][:3000]
        elements_with_ids = quiz_info.get('elements_with_ids', {})
        script_content = quiz_info.get('script_content', [])
        
        # Log what we're processing
        logger.debug(f"Page text length: {len(page_text)} chars")
        logger.debug(f"Elements with IDs: {len(elements_with_ids)}")
        logger.debug(f"Script tags: {len(script_content)}")
        
        # Build DYNAMIC context based on what we actually have
        context_parts = []
        
        # Part 1: Main page text
        context_parts.append(f"=== PAGE CONTENT ===\n{page_text}\n")
        
        # Part 2: Elements with IDs (if any)
        if elements_with_ids:
            logger.info(f"Available elements: {list(elements_with_ids.keys())}")
            elements_section = "=== RENDERED HTML ELEMENTS (after JavaScript execution) ===\n"
            for elem_id, elem_text in elements_with_ids.items():
                elements_section += f"\nElement ID: #{elem_id}\nContent: {elem_text[:500]}\n"
                logger.debug(f"#{elem_id} content: {elem_text[:300]}")
            context_parts.append(elements_section)
        
        # Part 3: Script content (if any)
        if script_content:
            scripts_section = "=== JAVASCRIPT CODE ON PAGE ===\n"
            for i, script in enumerate(script_content[:2], 1):  # First 2 scripts
                scripts_section += f"\n--- Script {i} ---\n{script[:1000]}\n"
            context_parts.append(scripts_section)
        
        # Combine all context
        full_context = "\n".join(context_parts)
        
        # Detect what type of question this is
        task_summary = task_info.get('task_summary', '').lower()
        is_meta_task = any(phrase in task_summary for phrase in ['anything you want', 'any value', 'whatever', 'your choice'])
        is_scraping_task = any(phrase in task_summary for phrase in ['scrape', 'extract', 'get the', 'find the'])
        has_elements = len(elements_with_ids) > 0
        has_scripts = len(script_content) > 0
        
        # Build dynamic instructions
        instructions = []
        
        if is_meta_task:
            instructions.append("✓ This quiz allows you to provide ANY value you want.")
            instructions.append("→ Simply respond with a value like: test")
        elif is_scraping_task and has_elements:
            instructions.append("✓ This is a SCRAPING task - extract information from the RENDERED page.")
            instructions.append(f"→ PRIORITY: Check 'RENDERED HTML ELEMENTS' section FIRST - there are {len(elements_with_ids)} element(s).")
            instructions.append(f"→ Available element IDs: {', '.join(elements_with_ids.keys())}")
            instructions.append("→ The content in these elements is what JavaScript generated/rendered on the page.")
            instructions.append("→ Extract the EXACT text from the relevant element (NOT from JavaScript code).")
            if has_scripts:
                instructions.append("→ Ignore base64 or raw code in JavaScript - use the DECODED content in elements.")
        elif is_scraping_task and has_scripts:
            instructions.append("✓ This is a SCRAPING task with JavaScript.")
            instructions.append("→ Check the 'JAVASCRIPT CODE' section for values that might be base64 encoded or hardcoded.")
            instructions.append("→ The secret might be in a variable, base64 string, or generated dynamically.")
        else:
            instructions.append("✓ Read the page content and answer the question directly.")
        
        instructions.append("\n⚠️ IMPORTANT: Respond with ONLY the answer value itself (no explanations, no formatting).")
        
        dynamic_instructions = "\n".join(instructions)
        full_context += f"\n\n=== INSTRUCTIONS FOR YOU ===\n{dynamic_instructions}\n"
        
        return full_context, is_meta_task, is_scraping_task
