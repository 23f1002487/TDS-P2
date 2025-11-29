"""
LLM Client Component
Handles all LLM interactions via OpenAI/AIPipe
"""
import json
import re
from typing import Dict, Any, List, Optional
from loguru import logger
from openai import AsyncOpenAI


class LLMClient:
    """
    Manages LLM interactions for task understanding, analysis, and direct questions
    """
    
    def __init__(self, aipipe_token: str, 
                 aipipe_base_url: str = "https://aipipe.org/openai/v1",
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        
        # OpenAI client setup with AIPipe
        self.client = AsyncOpenAI(
            api_key=aipipe_token,
            base_url=aipipe_base_url
        )
        
        logger.info(f"LLM Client initialized with model: {model_name}")
    
    async def understand_task(self, quiz_info: Dict) -> Dict[str, Any]:
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
3. Set "data_source_url" if there's:
   - A SEPARATE data file link (CSV, Excel, JSON, PDF with data)
   - A page/URL mentioned to SCRAPE (e.g., "Scrape /demo-scrape-data", "Get the secret from /page")
   - Any URL that contains data or information to extract (not the quiz page itself)
4. DO NOT confuse the quiz page URL or submission endpoint (/submit) with a data source
5. If instructions say "Scrape [URL]" or "Get [something] from [URL]", that URL is the data_source_url

Provide a detailed analysis in JSON format:
{{
    "task_summary": "Brief description of what needs to be done",
    "data_source_url": "URL of ACTUAL DATA FILE to download (leave empty if no data file)",
    "data_source_type": "pdf|csv|excel|json|html|image|api",
    "page_number": "specific page number if mentioned (for PDFs)",
    "target_column": "column name to analyze",
    "operation": "sum|average|count|max|min|filter|group|visualize|transcribe|other|meta_submission",
    "operation_details": "specific details about the operation",
    "expected_answer_type": "number|string|boolean|json|base64_image",
    "submit_url": "URL where answer should be submitted",
    "headers": {{"Header-Name": "Header-Value"}} (if API requires specific headers),
    "additional_instructions": "any other important details"
}}

Be precise. Extract exact URLs and column names from the instructions.
IMPORTANT: Respond with ONLY valid JSON, no other text."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
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
            logger.error(f"Error in task understanding: {e}")
            raise
    
    async def assisted_analysis(self, df, task_info: Dict, page_elements: Dict = None, table_name: str = "cleaned_data") -> Any:
        """Use LLM for complex data analysis"""
        logger.info("Using LLM for complex analysis")
        
        # Create prompt
        sample = df.head(10).to_string()
        
        # Add page elements info if available
        elements_info = ""
        if page_elements:
            elements_info = "\n=== PAGE ELEMENTS (e.g., cutoff values, thresholds) ===\n"
            for elem_id, elem_value in page_elements.items():
                elements_info += f"{elem_id}: {elem_value}\n"
        
        prompt = f"""Analyze this data and answer the question.

Task: {task_info.get('task_summary')}
Operation: {task_info.get('operation')}
Data columns: {list(df.columns)}
Data types: {df.dtypes.to_dict()}
Data shape: {df.shape}
Table name: {table_name}
{elements_info}
Sample data:
{sample}

IMPORTANT INSTRUCTIONS:
1. If page elements contain values like "cutoff", "threshold", use them in your analysis
2. For example, if cutoff=33364, sum only values ABOVE that cutoff (WHERE value_0 > 33364)
3. Use SQL to filter and aggregate the data efficiently
4. The table name is "{table_name}" - use this in your SQL query
5. Be precise with the WHERE condition - "above cutoff" means strictly greater than (>), not (>=)

Provide the answer in this JSON format:
{{
    "sql_query": "SQL query to get the answer using table '{table_name}'",
    "answer": "the final answer (numeric value)",
    "explanation": "brief explanation"
}}

Example SQL: SELECT SUM(value_0) FROM {table_name} WHERE value_0 > 33364

IMPORTANT: Respond with ONLY valid JSON."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            result_text = response.choices[0].message.content
            
            # Clean and parse
            result_text = re.sub(r'```json\n?', '', result_text)
            result_text = re.sub(r'```\n?', '', result_text)
            result_text = result_text.strip()
            
            analysis_result = json.loads(result_text)
            logger.success(f"LLM analysis completed")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in LLM-assisted analysis: {e}")
            raise
    
    async def answer_direct_question(self, context: str, task_info: Dict, 
                                    is_meta_task: bool = False, 
                                    is_scraping_task: bool = False) -> Any:
        """Answer a direct question without data analysis using LLM"""
        logger.info("Answering direct question with LLM")
        
        if is_scraping_task:
            prompt = f"""Extract the requested information from the page content below.

{context}

=== WHAT TO EXTRACT ===
Task: {task_info.get('task_summary')}
Operation: {task_info.get('operation')}
Additional Info: {task_info.get('additional_instructions', 'None')}

CRITICAL RULES:
1. Look for the ACTUAL VALUE in the page content - NOT placeholder text
2. Ignore any instructions about what to submit (like "your secret" or "the secret code you scraped")
3. Find the REAL secret/code/value that is displayed on the page
4. Return ONLY that exact value, nothing else
5. If you see HTML elements or rendered content, extract from there first

Your answer (the actual extracted value):"""
        else:
            prompt = f"""You are answering a quiz question. Analyze the content below and provide the answer.

{context}

=== TASK INFORMATION ===
Task: {task_info.get('task_summary')}
Operation: {task_info.get('operation')}
Additional Info: {task_info.get('additional_instructions', 'None')}

Your answer:"""
        
        try:
            # Dynamic system prompt based on task type
            if is_meta_task:
                system_prompt = "You provide simple direct answers. When asked for 'any value', give a simple value like 'test'."
            elif is_scraping_task:
                system_prompt = "You extract ACTUAL values from page content. NEVER return placeholder text like 'your secret' or 'the secret code you scraped'. Find and return the REAL value displayed on the page."
            else:
                system_prompt = "You answer questions concisely with just the answer value, no explanation."
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Try to parse as JSON if it looks like JSON
            if answer.startswith('{') or answer.startswith('['):
                try:
                    answer = json.loads(answer)
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Try to convert to number if it looks like a number
            elif answer.replace('.', '', 1).replace('-', '', 1).isdigit():
                try:
                    answer = float(answer) if '.' in answer else int(answer)
                except (ValueError, TypeError):
                    pass
            
            logger.success(f"Direct question answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering direct question: {e}")
            raise
    
    async def get_visualization_config(self, df, task_info: Dict) -> Dict[str, Any]:
        """Use LLM to determine best visualization"""
        logger.info("Getting visualization configuration from LLM")
        
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
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data visualization expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            result_text = response.choices[0].message.content
            
            result_text = re.sub(r'```json\n?', '', result_text)
            result_text = re.sub(r'```\n?', '', result_text)
            
            viz_info = json.loads(result_text)
            logger.success("Visualization config retrieved")
            
            return viz_info
        
        except Exception as e:
            logger.error(f"Error getting visualization config: {e}")
            raise

    async def raw_chat(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None) -> str:
        """Lightweight raw chat helper for flexible prompts (narratives, meta tasks).

        messages: list of {"role": "system|user|assistant", "content": str}
        system_prompt: optional system override; if omitted a neutral helper prompt is used
        temperature: optional override of default temperature

        Returns the assistant's final message content as a string.
        """
        logger.debug("raw_chat invoked with %d messages" % len(messages))

        temp = temperature if temperature is not None else self.temperature
        if system_prompt:
            # Prepend/merge system prompt if not already provided
            has_system = any(m.get("role") == "system" for m in messages)
            if not has_system:
                messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            # Ensure at least one system message for consistent behavior
            if not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": "You are a helpful assistant."}] + messages

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp
            )
            content = response.choices[0].message.content.strip()
            logger.success("raw_chat response received")
            return content
        except Exception as e:
            logger.error(f"raw_chat error: {e}")
            raise
