"""Report / Narrative Builder Component
Generates concise narrative summaries from analysis results using LLM.
"""
from loguru import logger
from typing import Dict, Any, Optional

class ReportBuilder:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def build(self, task_info: Dict, analysis_result: Any, sql: Optional[str] = None, chart_meta: Optional[Dict] = None) -> str:
        """Generate a narrative for the user summarizing approach and result."""
        prompt = f"""Create a concise narrative (max 120 words) summarizing the data task.
Task Summary: {task_info.get('task_summary')}
Operation: {task_info.get('operation')}
Result: {analysis_result}
SQL Used: {sql or 'None'}
Chart: {chart_meta.get('chart_type') if chart_meta else 'None'}
Important Instructions: {task_info.get('additional_instructions','None')}

Structure strictly as:
1. Objective
2. Method
3. Key Result
4. Confidence (High/Medium/Low)

Return ONLY plain text without markdown fencing."""
        try:
            messages = [
                {"role": "system", "content": "You are a senior data analyst. Respond plainly."},
                {"role": "user", "content": prompt}
            ]
            response = await self.llm.raw_chat(messages)
            text = response.strip()
            logger.success("Narrative generated")
            return {"text": text, "objective": task_info.get('task_summary', ''), "result": str(analysis_result)}
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return {"text": "Narrative unavailable", "error": str(e)}
