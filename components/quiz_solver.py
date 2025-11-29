"""
Refactored Quiz Solver using Modular Components
With LLMClient, WebScraper, and DataAnalyzer
"""
import asyncio
import json
from typing import Dict, Any
from urllib.parse import urlparse
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

import httpx

from . import LLMClient, WebScraper, DataAnalyzer
from .report_builder import ReportBuilder
from .capability_registry import CapabilityRegistry
from .transcriber import AudioTranscriber
from .vision import VisionExtractor

# Configure logger to write to production_log.log
logger.add("production_log.log", rotation="10 MB", retention="7 days", level="DEBUG")


class QuizSolver:
    """
    Production-grade quiz solver with modular components:
    - LLMClient for all LLM interactions
    - WebScraper for page fetching and parsing
    - DataAnalyzer for data processing and analysis
    """
    
    def __init__(self, email: str, secret: str, aipipe_token: str, 
                 aipipe_base_url: str = "https://aipipe.org/openai/v1",
                 model_name: str = "gpt-4o-mini"):
        self.email = email
        self.secret = secret
        
        # Initialize modular components
        self.llm_client = LLMClient(aipipe_token, aipipe_base_url, model_name)
        self.scraper = WebScraper()
        self.analyzer = DataAnalyzer()
        self.transcriber = AudioTranscriber(api_key=aipipe_token)
        self.vision = VisionExtractor()
        self.report_builder = ReportBuilder(self.llm_client)
        self.cap_registry = CapabilityRegistry()
        
        # HTTP client for submissions
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        logger.info(f"QuizSolver initialized with model: {model_name}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up all component resources"""
        await self.scraper.cleanup()
        await self.analyzer.cleanup()
        await self.http_client.aclose()
        logger.info("Cleanup complete")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def submit_answer(self, submit_url: str, quiz_url: str, answer: Any) -> Dict:
        """Submit answer with retry logic"""
        # Handle relative submit URLs
        if submit_url.startswith('/'):
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
        """Solve a single quiz using modular components"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Solving quiz: {quiz_url}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Step 1: Fetch and parse quiz page
            html_content = await self.scraper.fetch_page(quiz_url)
            quiz_info = self.scraper.parse_html(html_content)
            
            # Step 2: Understand task with LLM
            task_info = await self.llm_client.understand_task(quiz_info)
            self.cap_registry.record("operations", task_info.get('operation'))
            
            # Step 3: Handle audio transcription tasks separately
            if task_info.get('operation') == 'transcribe' or task_info.get('data_source_type') == 'audio':
                data_url = task_info.get('data_source_url')
                if data_url:
                    # Resolve relative URL
                    if not data_url.startswith('http://') and not data_url.startswith('https://'):
                        from urllib.parse import urljoin
                        data_url = urljoin(quiz_url, data_url)
                    
                    logger.info(f"Audio transcription task detected - downloading from: {data_url}")
                    audio_data = await self.analyzer.download_data(data_url, registry=self.cap_registry)
                    
                    # Get filename from URL
                    filename = data_url.split('/')[-1].split('?')[0]
                    if not filename or '.' not in filename:
                        filename = "audio.opus"
                    
                    # Transcribe
                    logger.info(f"Transcribing audio file: {filename}")
                    transcription = await self.transcriber.transcribe_bytes(audio_data, filename, registry=self.cap_registry)
                    logger.success(f"Transcription result: {transcription}")
                    
                    # Format and submit
                    formatted_answer = self.analyzer.format_answer(transcription, task_info.get('expected_answer_type', 'string'))
                    
                    # Generate narrative
                    narrative = None
                    try:
                        narrative = await self.report_builder.build(task_info, formatted_answer)
                        self.cap_registry.record("narrative", True)
                        logger.info(f"Narrative generated: {narrative.get('text','')[:80]}...")
                    except Exception as ne:
                        logger.warning(f"Narrative generation failed: {ne}")
                        narrative = {"text": "Narrative unavailable", "error": str(ne)}
                    
                    # Submit
                    submit_url = task_info.get('submit_url')
                    from urllib.parse import urlparse
                    if not submit_url or submit_url == quiz_url:
                        parsed = urlparse(quiz_url)
                        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                        logger.info(f"Using default submit endpoint: {submit_url}")
                    else:
                        parsed_submit = urlparse(submit_url)
                        submit_path = parsed_submit.path.rstrip('/')
                        if not submit_path.endswith('/submit') and ('project2' in submit_path or 'demo-' in submit_path or '?' in submit_url):
                            parsed = urlparse(quiz_url)
                            submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                            logger.info(f"Detected quiz page URL as submit_url, using default: {submit_url}")
                    
                    result = await self.submit_answer(submit_url, quiz_url, formatted_answer)
                    if isinstance(result, dict):
                        if narrative:
                            result['narrative'] = narrative
                        result['capabilities_used'] = self.cap_registry.snapshot()
                    logger.success(f"Quiz result: {result}")
                    return result
            
            # Step 3b: Handle image/vision tasks separately
            if task_info.get('data_source_type') == 'image':
                data_url = task_info.get('data_source_url')
                if data_url:
                    # Resolve relative URL
                    if not data_url.startswith('http://') and not data_url.startswith('https://'):
                        from urllib.parse import urljoin
                        data_url = urljoin(quiz_url, data_url)
                    
                    logger.info(f"Image/vision task detected - downloading from: {data_url}")
                    image_data = await self.analyzer.download_data(data_url, registry=self.cap_registry)
                    
                    # Check if this is a color analysis task (heatmap, color frequency, etc.)
                    task_lower = task_info.get('task_summary', '').lower()
                    is_color_task = any(keyword in task_lower for keyword in ['color', 'rgb', 'heatmap', 'hex', 'dominant'])
                    
                    if is_color_task:
                        # Use pixel color analysis instead of OCR
                        logger.info("Color analysis task detected - analyzing pixel colors")
                        answer = self.vision.get_dominant_color_hex(image_data, registry=self.cap_registry)
                        logger.success(f"Dominant color result: {answer}")
                    else:
                        # Perform OCR for text extraction
                        logger.info("Extracting text from image via OCR")
                        ocr_text = self.vision.ocr_bytes(image_data, registry=self.cap_registry)
                        logger.success(f"OCR result: {ocr_text[:200]}...")
                        
                        # Use LLM to analyze OCR text and answer the question
                        context = f"OCR extracted text from image:\n{ocr_text}\n\nTask: {task_info.get('task_summary')}"
                        answer = await self.llm_client.answer_direct_question(
                            context, task_info, is_meta_task=False, is_scraping_task=False
                        )
                    
                    # Format and submit
                    formatted_answer = self.analyzer.format_answer(answer, task_info.get('expected_answer_type', 'string'))
                    
                    # Generate narrative
                    narrative = None
                    try:
                        narrative = await self.report_builder.build(task_info, formatted_answer)
                        self.cap_registry.record("narrative", True)
                        logger.info(f"Narrative generated: {narrative.get('text','')[:80]}...")
                    except Exception as ne:
                        logger.warning(f"Narrative generation failed: {ne}")
                        narrative = {"text": "Narrative unavailable", "error": str(ne)}
                    
                    # Submit
                    submit_url = task_info.get('submit_url')
                    from urllib.parse import urlparse
                    if not submit_url or submit_url == quiz_url:
                        parsed = urlparse(quiz_url)
                        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                        logger.info(f"Using default submit endpoint: {submit_url}")
                    else:
                        parsed_submit = urlparse(submit_url)
                        submit_path = parsed_submit.path.rstrip('/')
                        if not submit_path.endswith('/submit') and ('project2' in submit_path or 'demo-' in submit_path or '?' in submit_url):
                            parsed = urlparse(quiz_url)
                            submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                            logger.info(f"Detected quiz page URL as submit_url, using default: {submit_url}")
                    
                    result = await self.submit_answer(submit_url, quiz_url, formatted_answer)
                    if isinstance(result, dict):
                        if narrative:
                            result['narrative'] = narrative
                        result['capabilities_used'] = self.cap_registry.snapshot()
                    logger.success(f"Quiz result: {result}")
                    return result
            
            # Step 4: Download and process data (if data source exists)
            df, table_name = await self.analyzer.process_data(task_info, quiz_url, registry=self.cap_registry)
            
            # Step 4b: Handle data transformation tasks (CSV to JSON, etc.)
            if df is not None and task_info.get('operation') == 'transform_to_json':
                logger.info("Data transformation task detected - converting to JSON array")
                # Convert DataFrame to JSON array format
                import json
                json_array = json.loads(df.to_json(orient='records'))
                
                # Sort by id if present
                if 'id' in df.columns:
                    json_array = sorted(json_array, key=lambda x: x.get('id', 0))
                
                # Format as JSON string
                formatted_answer = json.dumps(json_array)
                logger.success(f"Transformed data to JSON array with {len(json_array)} records")
                
                # Generate narrative
                narrative = None
                try:
                    narrative = await self.report_builder.build(task_info, formatted_answer)
                    self.cap_registry.record("narrative", True)
                    logger.info(f"Narrative generated: {narrative.get('text','')[:80]}...")
                except Exception as ne:
                    logger.warning(f"Narrative generation failed: {ne}")
                    narrative = {"text": "Narrative unavailable", "error": str(ne)}
                
                # Submit
                submit_url = task_info.get('submit_url')
                from urllib.parse import urlparse
                if not submit_url or submit_url == quiz_url:
                    parsed = urlparse(quiz_url)
                    submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                    logger.info(f"Using default submit endpoint: {submit_url}")
                else:
                    parsed_submit = urlparse(submit_url)
                    submit_path = parsed_submit.path.rstrip('/')
                    if not submit_path.endswith('/submit') and ('project2' in submit_path or 'demo-' in submit_path or '?' in submit_url):
                        parsed = urlparse(quiz_url)
                        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                        logger.info(f"Detected quiz page URL as submit_url, using default: {submit_url}")
                
                result = await self.submit_answer(submit_url, quiz_url, formatted_answer)
                if isinstance(result, dict):
                    if narrative:
                        result['narrative'] = narrative
                    result['capabilities_used'] = self.cap_registry.snapshot()
                logger.success(f"Quiz result: {result}")
                return result
            
            # Step 5: Perform analysis or answer directly
            if df is not None and isinstance(df, tuple) and df[0] == 'HTML_SCRAPING':
                # This is an HTML scraping task - use Playwright to fetch with JS execution
                _, scraping_url, _ = df
                logger.info(f"HTML scraping task detected - using Playwright to fetch: {scraping_url}")
                self.cap_registry.record("html_scraping", True)
                
                # Use Playwright to fetch and render the page (executes JavaScript)
                scraped_html = await self.scraper.fetch_page(scraping_url)
                scraped_info = self.scraper.parse_html(scraped_html)
                
                # Build context from the scraped page (after JS execution)
                context, is_meta_task, is_scraping_task = self.scraper.build_dynamic_context(
                    scraped_info, task_info
                )
                
                # Ask LLM to extract the specific value from scraped content
                logger.info("Extracting value from scraped page (after JS execution)")
                answer = await self.llm_client.answer_direct_question(
                    context, task_info, is_meta_task=False, is_scraping_task=True
                )
            elif df is not None and table_name is not None:
                # Data-based question: analyze the data
                logger.info("Data available - performing data analysis")
                # Pass quiz_info to analysis so it has access to page elements (like cutoff values)
                answer = await self.analyzer.perform_analysis(
                    df, table_name, task_info, self.llm_client, quiz_info
                )
                self.cap_registry.record("data_analysis", True)
            else:
                # Direct question without data: ask LLM directly
                logger.info("No data available - answering question directly with LLM")
                context, is_meta_task, is_scraping_task = self.scraper.build_dynamic_context(
                    quiz_info, task_info
                )
                answer = await self.llm_client.answer_direct_question(
                    context, task_info, is_meta_task, is_scraping_task
                )
                self.cap_registry.record("direct_answer", True)
            
            # Step 5: Format answer
            formatted_answer = self.analyzer.format_answer(
                answer,
                task_info.get('expected_answer_type', 'string')
            )

            # Step 5b: Generate narrative/explanation (non-blocking if fails)
            narrative = None
            try:
                narrative = await self.report_builder.build(task_info, answer)
                self.cap_registry.record("narrative", True)
                if isinstance(narrative, dict):
                    logger.info(f"Narrative generated: {narrative.get('text','')[:80]}...")
                else:
                    logger.info(f"Narrative generated: {str(narrative)[:80]}...")
            except Exception as ne:
                logger.warning(f"Narrative generation failed: {ne}")
                narrative = {"text": "Narrative unavailable", "error": str(ne)}
                self.cap_registry.record("narrative_failed", True)
            
            # Step 7: Submit answer
            submit_url = task_info.get('submit_url')
            # Validate submit_url: default to /submit if not provided or if it looks like a quiz page
            from urllib.parse import urlparse
            if not submit_url or submit_url == quiz_url:
                # No submit_url or it's the same as quiz_url - use default /submit
                parsed = urlparse(quiz_url)
                submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                logger.info(f"Using default submit endpoint: {submit_url}")
            else:
                # Check if submit_url is actually a quiz page path (not /submit)
                parsed_submit = urlparse(submit_url)
                submit_path = parsed_submit.path.rstrip('/')
                # If path doesn't end with /submit and looks like a quiz page, fix it
                if not submit_path.endswith('/submit') and ('project2' in submit_path or 'demo-' in submit_path or '?' in submit_url):
                    parsed = urlparse(quiz_url)
                    submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                    logger.info(f"Detected quiz page URL as submit_url, using default: {submit_url}")
            result = await self.submit_answer(submit_url, quiz_url, formatted_answer)

            # Attach narrative & capabilities to result for downstream consumers
            if isinstance(result, dict):
                if narrative:
                    result['narrative'] = narrative
                result['capabilities_used'] = self.cap_registry.snapshot()
            
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
        max_retries_per_quiz = 3
        
        # Track statistics
        stats = {
            'total_tasks': 0,
            'correct_tasks': 0,
            'wrong_tasks': 0,
            'tasks': []  # List of task details
        }
        
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
                
                # Try solving with retries
                quiz_solved = False
                task_attempts = 0
                for attempt in range(max_retries_per_quiz):
                    # Check if we still have time
                    current_elapsed = time.time() - start_time
                    if current_elapsed > 175:
                        logger.warning("Time limit reached, cannot retry")
                        break
                    
                    try:
                        task_attempts += 1
                        stats['total_tasks'] += 1
                        task_start = time.time()
                        
                        result = await self.solve_single_quiz(current_url)
                        
                        task_time = time.time() - task_start
                        
                        # Check the response fields
                        is_correct = result.get('correct', False)
                        reason = result.get('reason', '')
                        next_url = result.get('url', '')
                        
                        # Track task result
                        task_info = {
                            'quiz_num': quiz_count,
                            'attempt': attempt + 1,
                            'correct': is_correct,
                            'reason': reason,
                            'time_taken': task_time,
                            'url': current_url
                        }
                        stats['tasks'].append(task_info)
                        
                        if is_correct:
                            stats['correct_tasks'] += 1
                            logger.success(f"✓ Quiz #{quiz_count} CORRECT!")
                            if reason:
                                logger.info(f"Success reason: {reason}")
                            current_url = next_url
                            quiz_solved = True
                            break
                        else:
                            stats['wrong_tasks'] += 1
                            logger.warning(f"✗ Quiz #{quiz_count} INCORRECT (attempt {attempt + 1}/{max_retries_per_quiz})")
                            logger.warning(f"Reason: {reason}")
                            
                            # Check if we should retry based on time limit
                            time_since_quiz_start = time.time() - quiz_start_time
                            if time_since_quiz_start < 170:
                                if attempt < max_retries_per_quiz - 1:
                                    logger.info(f"Retrying quiz (time since quiz start: {time_since_quiz_start:.1f}s)...")
                                    await asyncio.sleep(1)
                                    continue
                            
                            # Move to next quiz URL if provided
                            if next_url:
                                logger.info(f"Moving to next quiz URL: {next_url}")
                                current_url = next_url
                            break
                    
                    except Exception as e:
                        logger.exception(f"Attempt {attempt + 1} failed with exception: {e}")
                        
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
            
            # Calculate score (correct tasks / total tasks)
            score_pct = (stats['correct_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0
            
            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info(f"QUIZ CHAIN SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total Time:      {total_time:.1f}s")
            logger.info(f"Quizzes:         {quiz_count}")
            logger.info(f"Total Attempts:  {stats['total_tasks']}")
            logger.info(f"Correct:         {stats['correct_tasks']} ✓")
            logger.info(f"Wrong:           {stats['wrong_tasks']} ✗")
            logger.info(f"Score:           {score_pct:.1f}% ({stats['correct_tasks']}/{stats['total_tasks']})")
            # Capability summary logging
            logger.info(f"Capabilities Used: {json.dumps(self.cap_registry.snapshot(), indent=2)}")
            logger.info(f"{'='*60}")
            
            # Print per-quiz breakdown
            if stats['tasks']:
                logger.info("\nPer-Quiz Breakdown:")
                logger.info(f"{'─'*60}")
                current_quiz = None
                for task in stats['tasks']:
                    if task['quiz_num'] != current_quiz:
                        current_quiz = task['quiz_num']
                        logger.info(f"\n📋 Quiz #{task['quiz_num']}:")
                    
                    status = "✓ CORRECT" if task['correct'] else "✗ WRONG"
                    logger.info(f"  Attempt {task['attempt']}: {status} ({task['time_taken']:.1f}s) - {task['reason']}")
                logger.info(f"{'─'*60}\n")
            
        finally:
            # Cleanup
            await self.cleanup()
