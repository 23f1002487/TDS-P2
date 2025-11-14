"""
Ultimate Enhanced Quiz Solver
With Playwright, OpenAI (via AIPipe), Tenacity, and Loguru
"""
import asyncio
import json
import re
from typing import Optional, Dict, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import httpx
from playwright.async_api import async_playwright, Page, Browser
from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from data_processor import EnhancedDataProcessor
from visualization import Visualizer


class QuizSolver:
    """
    Production-grade quiz solver with:
    - Playwright for reliable JS rendering
    - OpenAI (via AIPipe) for LLM inference
    - Tenacity for retry logic  
    - Loguru for better logging
    - httpx for async HTTP
    """
    
    def __init__(self, email: str, secret: str, aipipe_token: str, 
                 aipipe_base_url: str = "https://aipipe.org/openai/v1",
                 model_name: str = "gpt-4o-mini"):
        self.email = email
        self.secret = secret
        self.aipipe_token = aipipe_token
        self.model_name = model_name
        
        # Initialize components
        self.data_processor = EnhancedDataProcessor()
        self.visualizer = Visualizer()
        
        # OpenAI client setup with AIPipe
        self.llm_client = AsyncOpenAI(
            api_key=aipipe_token,
            base_url=aipipe_base_url
        )
        
        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Playwright browser (initialized on demand)
        self.browser: Optional[Browser] = None
        self.playwright = None
        
        logger.info(f"QuizSolver initialized with model: {model_name}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up browser and HTTP client"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        await self.http_client.aclose()
        logger.info("Cleanup complete")
    
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
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
        reraise=True
    )
    async def fetch_quiz_page(self, url: str) -> str:
        """
        Fetch and render JavaScript-based quiz page using Playwright
        Auto-retries with exponential backoff
        """
        logger.info(f"Fetching quiz page: {url}")
        
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
            
            logger.success(f"Quiz page fetched successfully: {len(html_content)} bytes")
            return html_content
            
        except Exception as e:
            logger.error(f"Error fetching quiz page: {e}")
            raise
        finally:
            await page.close()
    
    def parse_quiz_instructions(self, html_content: str) -> Dict[str, Any]:
        """Extract quiz instructions from HTML"""
        logger.info("Parsing quiz instructions")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to find result div first
        result_div = soup.find(id='result')
        if result_div:
            text = result_div.get_text(separator='\n', strip=True)
            result_html = str(result_div)
            logger.info("Found quiz in #result div")
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
        
        logger.info(f"Extracted {len(elements_with_ids)} elements with IDs")
        
        return {
            'text': text,
            'links': links,
            'html': html_content,
            'result_html': result_html,  # HTML of just the result div (or full if no result div)
            'elements_with_ids': elements_with_ids  # Dictionary of id -> text content
        }
    
    async def understand_task_with_langchain(self, quiz_info: Dict) -> Dict[str, Any]:
        """Use LLM to understand the quiz task"""
        logger.info("Analyzing quiz task with LLM")
        
        prompt = f"""You are an expert data analyst. Analyze this quiz task and extract key information.

Quiz Instructions:
{quiz_info['text']}

Available Links:
{json.dumps(quiz_info['links'], indent=2)}

IMPORTANT RULES:
1. If the instructions say to "POST this JSON" or show a JSON template to submit, this is a META-TASK about submission format
2. Look for phrases like "anything you want", "any value", or placeholders - these mean NO data analysis is needed
3. Only set "data_source_url" if there's a SEPARATE data file link (CSV, Excel, JSON data file, PDF with data)
4. DO NOT confuse the quiz page URL or submission instructions with a data source

Provide a detailed analysis in JSON format:
{{
    "task_summary": "Brief description of what needs to be done",
    "data_source_url": "URL of ACTUAL DATA FILE to download (leave empty if no data file)",
    "data_source_type": "pdf|csv|excel|json|html|image",
    "page_number": "specific page number if mentioned (for PDFs)",
    "target_column": "column name to analyze",
    "operation": "sum|average|count|max|min|filter|group|visualize|other|meta_submission",
    "operation_details": "specific details about the operation",
    "expected_answer_type": "number|string|boolean|json|base64_image",
    "submit_url": "URL where answer should be submitted",
    "additional_instructions": "any other important details"
}}

Be precise. Extract exact URLs and column names from the instructions.
IMPORTANT: Respond with ONLY valid JSON, no other text."""
        
        try:
            # Call OpenAI API via AIPipe
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            
            # Clean and parse response
            result_text = re.sub(r'```json\n?', '', result_text)
            result_text = re.sub(r'```\n?', '', result_text)
            result_text = result_text.strip()
            
            task_info = json.loads(result_text)
            logger.success(f"Task understood: {task_info['task_summary']}")
            
            return task_info
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {result_text}")
            raise
        except Exception as e:
            logger.error(f"Error in LangChain task understanding: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def download_data(self, url: str) -> bytes:
        """Download data file with retry logic"""
        logger.info(f"Downloading data from: {url}")
        
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            data = response.content
            logger.success(f"Downloaded {len(data)} bytes")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    async def process_data(self, task_info: Dict) -> tuple:
        """Download and process data based on task information"""
        logger.info("Processing data")
        
        data_url = task_info.get('data_source_url')
        if not data_url:
            logger.warning("No data source URL found")
            return None, None
        
        # Download data
        data_content = await self.download_data(data_url)
        
        # Check if downloaded content is HTML (quiz page instead of actual data)
        content_str = data_content.decode('utf-8', errors='ignore')[:200].strip()
        if content_str.startswith('<') or content_str.startswith('<!DOCTYPE'):
            logger.warning("Downloaded content appears to be HTML, not data. Skipping data processing.")
            logger.info(f"Content preview: {content_str[:100]}")
            return None, None
        
        # Build kwargs for loading
        kwargs = {}
        if task_info.get('page_number'):
            try:
                page_num = int(task_info['page_number'])
                kwargs['page_numbers'] = [page_num]
            except:
                pass
        
        # Save to temporary file for processing
        import tempfile
        import os
        
        # Extract file extension safely (avoid getting domain parts)
        url_parts = data_url.split('/')[-1].split('?')[0]  # Get filename without query params
        valid_extensions = {'csv', 'xlsx', 'xls', 'json', 'pdf', 'parquet', 'html', 'xml', 'tsv', 'txt'}
        
        if '.' in url_parts:
            file_ext = url_parts.split('.')[-1].lower()
            if file_ext not in valid_extensions:
                # Invalid extension, use data_source_type or default
                type_map = {'csv': 'csv', 'excel': 'xlsx', 'json': 'json', 'pdf': 'pdf'}
                file_ext = type_map.get(task_info.get('data_source_type', '').lower(), 'dat')
        else:
            # No extension in URL
            type_map = {'csv': 'csv', 'excel': 'xlsx', 'json': 'json', 'pdf': 'pdf'}
            file_ext = type_map.get(task_info.get('data_source_type', '').lower(), 'dat')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
            tmp.write(data_content)
            tmp_path = tmp.name
        
        try:
            # Use the complete data pipeline
            df, table_name = self.data_processor.complete_data_pipeline(
                tmp_path,
                source_type=task_info.get('data_source_type', 'auto'),
                **kwargs
            )
            
            logger.success(f"Data processed: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df, table_name
            
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
    
    async def perform_analysis(self, df, table_name: str, task_info: Dict) -> Any:
        """Perform the required analysis using DuckDB"""
        logger.info("Performing analysis")
        
        operation = task_info.get('operation', '').lower()
        target_column = task_info.get('target_column')
        
        # Find target column (case-insensitive, fuzzy match)
        if target_column:
            target_column_clean = target_column.lower().strip()
            for col in df.columns:
                if col.lower() == target_column_clean or target_column_clean in col.lower():
                    target_column = col
                    break
        
        logger.info(f"Operation: {operation}, Target column: {target_column}")
        
        try:
            if operation == 'sum' and target_column:
                result = self.data_processor.query(
                    f"SELECT SUM({target_column}) as result FROM {table_name}"
                )['result'].iloc[0]
                logger.success(f"Sum result: {result}")
                return result
            
            elif operation == 'average' and target_column:
                result = self.data_processor.query(
                    f"SELECT AVG({target_column}) as result FROM {table_name}"
                )['result'].iloc[0]
                logger.success(f"Average result: {result}")
                return result
            
            elif operation == 'count':
                if target_column:
                    sql = f"SELECT COUNT({target_column}) as result FROM {table_name}"
                else:
                    sql = f"SELECT COUNT(*) as result FROM {table_name}"
                result = self.data_processor.query(sql)['result'].iloc[0]
                logger.success(f"Count result: {result}")
                return result
            
            elif operation in ['max', 'min'] and target_column:
                func = operation.upper()
                result = self.data_processor.query(
                    f"SELECT {func}({target_column}) as result FROM {table_name}"
                )['result'].iloc[0]
                logger.success(f"{func} result: {result}")
                return result
            
            elif operation == 'visualize':
                return await self._create_visualization(df, task_info)
            
            else:
                # Use LLM for complex analysis
                return await self._llm_assisted_analysis(df, table_name, task_info)
        
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            # Fallback to LLM
            return await self._llm_assisted_analysis(df, table_name, task_info)
    
    async def _llm_assisted_analysis(self, df, table_name: str, task_info: Dict) -> Any:
        """Use LLM for complex analysis"""
        logger.info("Using LLM for complex analysis")
        
        # Create prompt
        sample = df.head(10).to_string()
        
        prompt = f"""Analyze this data and answer the question.

Task: {task_info.get('task_summary')}
Operation: {task_info.get('operation')}
Data columns: {list(df.columns)}
Data types: {df.dtypes.to_dict()}
Data shape: {df.shape}

Sample data:
{sample}

Provide the answer in this JSON format:
{{
    "sql_query": "SQL query to get the answer (if applicable)",
    "answer": "the final answer",
    "explanation": "brief explanation"
}}

If SQL is not applicable, compute the answer directly.
IMPORTANT: Respond with ONLY valid JSON."""
        
        try:
            # Call OpenAI API via AIPipe
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            
            # Clean and parse
            result_text = re.sub(r'```json\n?', '', result_text)
            result_text = re.sub(r'```\n?', '', result_text)
            result_text = result_text.strip()
            
            analysis_result = json.loads(result_text)
            
            # Execute SQL if provided
            if analysis_result.get('sql_query'):
                try:
                    result_df = self.data_processor.query(analysis_result['sql_query'])
                    logger.info(f"SQL query result:\n{result_df}")
                    
                    if len(result_df) == 1 and len(result_df.columns) == 1:
                        answer = result_df.iloc[0, 0]
                    else:
                        answer = result_df.to_dict('records')
                except Exception as e:
                    logger.error(f"SQL execution failed: {e}")
                    answer = analysis_result.get('answer')
            else:
                answer = analysis_result.get('answer')
            
            logger.success(f"LLM analysis result: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Error in LLM-assisted analysis: {e}")
            raise
    
    async def _answer_direct_question(self, quiz_info: Dict, task_info: Dict) -> Any:
        """Answer a direct question without data analysis using LLM"""
        logger.info("Answering direct question with LLM")
        
        # Get all available information
        page_text = quiz_info['text'][:3000]
        elements_with_ids = quiz_info.get('elements_with_ids', {})
        
        # Build comprehensive context
        elements_info = "\n".join([f"- Element #{elem_id}: {elem_text[:200]}" for elem_id, elem_text in elements_with_ids.items()])
        
        prompt = f"""You are answering a quiz question. Read the page content carefully and extract or compute the answer.

Quiz Page Content (Text):
{page_text}

HTML Elements with IDs (may contain secrets/codes):
{elements_info if elements_info else "No specific elements found"}

Task: {task_info.get('task_summary')}
Operation: {task_info.get('operation')}
Additional Instructions: {task_info.get('additional_instructions', '')}

INSTRUCTIONS:
1. If the quiz asks you to "scrape" or "extract" information (like a secret code, password, or specific text):
   - Look for that information in the Quiz Page Content above
   - Extract the EXACT value shown on the page
   - Do NOT make up or generate placeholder values

2. If the quiz says you can provide "anything you want" or "any value":
   - You can provide any reasonable value like "test", "hello", or 42

3. If it's a calculation or data analysis question:
   - Perform the calculation based on information in the page
   - Return the computed result

4. If it's asking for information shown on the page:
   - Read and extract that specific information
   - Return it exactly as shown

Respond with ONLY the answer value itself (no explanations, no markdown, no quotes unless the answer itself contains quotes).

Examples:
- If a secret code "ABC123" is shown: ABC123
- If asked to provide any value: test
- If calculating 2+2: 4
- If extracting a name "John Smith": John Smith

Your answer:"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions concisely. Provide only the answer without explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Try to parse as JSON if it looks like JSON
            if answer.startswith('{') or answer.startswith('['):
                try:
                    answer = json.loads(answer)
                except:
                    pass
            
            # Try to convert to number if it looks like a number
            elif answer.replace('.', '', 1).replace('-', '', 1).isdigit():
                try:
                    answer = float(answer) if '.' in answer else int(answer)
                except:
                    pass
            
            logger.success(f"Direct question answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering direct question: {e}")
            raise
    
    async def _create_visualization(self, df, task_info: Dict) -> str:
        """Create visualization based on task requirements"""
        logger.info("Creating visualization")
        
        # Use LLM to determine best visualization
        prompt = f"""Based on this task, what type of chart should be created?

Task: {task_info.get('task_summary')}
Columns: {list(df.columns)}

Choose from: bar_chart, line_chart, scatter_plot, pie_chart, histogram, heatmap

Respond with JSON:
{{
    "chart_type": "type",
    "x_column": "column for x-axis",
    "y_column": "column for y-axis",
    "title": "chart title"
}}

IMPORTANT: Respond with ONLY valid JSON."""
        
        try:
            # Call OpenAI API via AIPipe
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            result_text = re.sub(r'```json\n?', '', result_text)
            result_text = re.sub(r'```\n?', '', result_text)
            
            viz_info = json.loads(result_text)
            
            chart_type = viz_info.get('chart_type')
            x_col = viz_info.get('x_column')
            y_col = viz_info.get('y_column')
            title = viz_info.get('title', 'Chart')
            
            # Create chart
            if chart_type == 'bar_chart':
                result = self.visualizer.create_bar_chart(df, x_col, y_col, title)
            elif chart_type == 'line_chart':
                result = self.visualizer.create_line_chart(df, x_col, y_col, title)
            elif chart_type == 'scatter_plot':
                result = self.visualizer.create_scatter_plot(df, x_col, y_col, title)
            elif chart_type == 'pie_chart':
                result = self.visualizer.create_pie_chart(df, x_col, y_col, title)
            else:
                result = self.visualizer.create_bar_chart(df, x_col, y_col, title)
            
            logger.success("Visualization created")
            return result
        
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            raise
    
    def format_answer(self, answer: Any, expected_type: str) -> Any:
        """Format answer according to expected type"""
        logger.info(f"Formatting answer as: {expected_type}")
        
        if expected_type == 'number':
            try:
                if isinstance(answer, (int, float)):
                    return int(answer) if answer == int(answer) else float(answer)
                return float(answer)
            except:
                return answer
        
        elif expected_type == 'boolean':
            if isinstance(answer, bool):
                return answer
            if isinstance(answer, str):
                return answer.lower() in ['true', 'yes', '1', 't', 'y']
            return bool(answer)
        
        elif expected_type == 'json':
            if isinstance(answer, (dict, list)):
                return answer
            try:
                return json.loads(answer)
            except:
                return answer
        
        elif expected_type == 'base64_image':
            return answer
        
        else:
            return str(answer)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def submit_answer(self, submit_url: str, quiz_url: str, answer: Any) -> Dict:
        """Submit answer with retry logic"""
        # Handle relative submit URLs
        if submit_url.startswith('/'):
            from urllib.parse import urlparse
            parsed_quiz = urlparse(quiz_url)
            submit_url = f"{parsed_quiz.scheme}://{parsed_quiz.netloc}{submit_url}"
            logger.info(f"Converted relative URL to: {submit_url}")
        
        payload = {
            "email": self.email,
            "secret": self.secret,
            "url": quiz_url,
            "answer": answer
        }
        
        logger.info(f"Submitting answer to: {submit_url}")
        logger.info(f"Answer: {answer}")
        
        try:
            response = await self.http_client.post(
                submit_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Submit response: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting answer: {e}")
            raise
    
    async def solve_single_quiz(self, quiz_url: str) -> Dict:
        """Solve a single quiz"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Solving quiz: {quiz_url}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Step 1: Fetch quiz page
            html_content = await self.fetch_quiz_page(quiz_url)
            
            # Step 2: Parse instructions
            quiz_info = self.parse_quiz_instructions(html_content)
            
            # Step 3: Understand task with LLM
            task_info = await self.understand_task_with_langchain(quiz_info)
            
            # Step 4: Download and process data (if data source exists)
            df, table_name = await self.process_data(task_info)
            
            # Step 5: Perform analysis
            if df is not None and table_name is not None:
                # Data-based question: analyze the data
                logger.info("Data available - performing data analysis")
                answer = await self.perform_analysis(df, table_name, task_info)
            else:
                # Direct question without data: ask LLM directly
                logger.info("No data available - answering question directly with LLM")
                answer = await self._answer_direct_question(quiz_info, task_info)
            
            # Step 6: Format answer
            formatted_answer = self.format_answer(
                answer,
                task_info.get('expected_answer_type', 'string')
            )
            
            # Step 7: Submit answer
            submit_url = task_info.get('submit_url')
            result = await self.submit_answer(submit_url, quiz_url, formatted_answer)
            
            logger.success(f"Quiz result: {result}")
            return result
            
        except Exception as e:
            logger.exception(f"Error solving quiz: {e}")
            raise
    
    def solve_quiz_chain_sync(self, initial_url: str):
        """Synchronous wrapper for async solve_quiz_chain"""
        asyncio.run(self.solve_quiz_chain(initial_url))
    
    async def solve_quiz_chain(self, initial_url: str):
        """Solve a chain of quizzes with retry logic"""
        import time
        start_time = time.time()
        current_url = initial_url
        quiz_count = 0
        max_quizzes = 20
        max_retries_per_quiz = 3  # Allow up to 3 attempts per quiz
        
        try:
            while current_url and quiz_count < max_quizzes:
                quiz_count += 1
                quiz_start_time = time.time()
                elapsed = quiz_start_time - start_time
                
                # Check 3-minute timeout
                if elapsed > 175:
                    logger.warning("Approaching 3-minute timeout limit, stopping")
                    break
                
                logger.info(f"\n{'#'*60}")
                logger.info(f"QUIZ #{quiz_count}")
                logger.info(f"URL: {current_url}")
                logger.info(f"Elapsed: {elapsed:.1f}s / 180s")
                logger.info(f"{'#'*60}\n")
                
                # Try solving with retries (only if within time limit)
                quiz_solved = False
                for attempt in range(max_retries_per_quiz):
                    # Check if we still have time (within 3 minutes from start)
                    current_elapsed = time.time() - start_time
                    if current_elapsed > 175:
                        logger.warning("Time limit reached, cannot retry")
                        break
                    
                    try:
                        result = await self.solve_single_quiz(current_url)
                        
                        # Check the response fields
                        is_correct = result.get('correct', False)
                        reason = result.get('reason', '')
                        next_url = result.get('url', '')
                        
                        if is_correct:
                            logger.success(f"✓ Quiz #{quiz_count} CORRECT!")
                            if reason:
                                logger.info(f"Success reason: {reason}")
                            current_url = next_url
                            quiz_solved = True
                            break
                        else:
                            logger.warning(f"✗ Quiz #{quiz_count} INCORRECT (attempt {attempt + 1}/{max_retries_per_quiz})")
                            logger.warning(f"Reason: {reason}")
                            
                            # Check if we should retry based on time limit
                            time_since_quiz_start = time.time() - quiz_start_time
                            if time_since_quiz_start < 170:  # Allow retry if quiz started less than 170s ago
                                if attempt < max_retries_per_quiz - 1:
                                    logger.info(f"Retrying quiz (time since quiz start: {time_since_quiz_start:.1f}s)...")
                                    await asyncio.sleep(1)  # Brief pause before retry
                                    continue
                            
                            # If we can't retry or last attempt, move to next quiz URL if provided
                            if next_url:
                                logger.info(f"Moving to next quiz URL: {next_url}")
                                current_url = next_url
                            break
                    
                    except Exception as e:
                        logger.exception(f"Attempt {attempt + 1} failed with exception: {e}")
                        
                        # Check if we should retry based on time
                        current_elapsed = time.time() - start_time
                        if current_elapsed > 175 or attempt == max_retries_per_quiz - 1:
                            logger.error("Cannot retry - time limit or max attempts reached")
                            return
                        
                        logger.info("Retrying after exception...")
                        await asyncio.sleep(2)
                
                # If quiz wasn't solved and no next URL, stop
                if not quiz_solved and not current_url:
                    logger.warning("No more quizzes to solve")
                    break
            
            total_time = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"QUIZ CHAIN COMPLETE")
            logger.info(f"Solved {quiz_count} quizzes in {total_time:.1f}s")
            logger.info(f"{'='*60}\n")
            
        finally:
            # Cleanup
            await self.cleanup()
