# Implementation Guide

This guide walks you through implementing the complete quiz solver.

## Phase 1: Setup and Testing (Day 1)

### 1.1 Initial Setup
```bash
./quickstart.sh
```

### 1.2 Configure Environment
Edit `.env`:
```bash
STUDENT_EMAIL=23f1002487@ds.study.iitm.ac.in
STUDENT_SECRET=this-is-agni
OPENAI_API_KEY=sk-your-actual-key-here
PORT=7860
```

### 1.3 Test Locally
```bash
# Terminal 1: Start server
python app.py

# Terminal 2: Run tests
python -m tests.test_api
```

## Phase 2: Complete Quiz Solver Logic (Days 2-3)

### 2.1 Update `quiz_solver.py`

The current `solve_with_llm()` method needs to be enhanced. Here's what to add:

```python
def solve_with_llm(self, quiz_text, quiz_url):
    """Enhanced LLM solving with actual implementation"""
    
    # Step 1: Understand the task
    understanding_prompt = f"""Analyze this quiz task and break it down:

Quiz: {quiz_text}
URL: {quiz_url}

Provide a JSON response:
{{
    "task_type": "data_extraction|analysis|visualization",
    "data_source": "URL of data file",
    "required_operations": ["operation1", "operation2"],
    "expected_answer_type": "number|string|boolean|json|base64_image",
    "submit_url": "URL to submit answer"
}}
"""
    
    # Step 2: Get task understanding
    understanding = self._call_llm(understanding_prompt)
    task_info = json.loads(understanding)
    
    # Step 3: Download and process data
    if task_info.get('data_source'):
        data = self._download_data(task_info['data_source'])
        processed_data = self._process_data(data, task_info)
    
    # Step 4: Perform analysis
    answer = self._perform_analysis(processed_data, task_info)
    
    # Step 5: Format answer
    formatted_answer = self._format_answer(answer, task_info['expected_answer_type'])
    
    return {
        'answer': formatted_answer,
        'submit_url': task_info['submit_url']
    }
```

### 2.2 Implement Helper Methods

Add these methods to `QuizSolver`:

```python
def _download_data(self, url):
    """Download data file"""
    from data_processor import DataProcessor
    return DataProcessor.download_file(url)

def _process_data(self, data, task_info):
    """Process data based on file type"""
    from data_processor import DataProcessor
    
    # Detect file type and process accordingly
    if 'pdf' in task_info.get('data_source', '').lower():
        return DataProcessor.extract_pdf_tables(data)
    elif 'csv' in task_info.get('data_source', '').lower():
        return DataProcessor.process_csv(data)
    # Add more file types...

def _perform_analysis(self, data, task_info):
    """Perform required analysis"""
    # Use pandas for data operations
    # Call LLM for complex analysis if needed
    pass

def _format_answer(self, answer, answer_type):
    """Format answer based on expected type"""
    if answer_type == 'number':
        return float(answer) if '.' in str(answer) else int(answer)
    elif answer_type == 'base64_image':
        # Convert image to base64
        from visualization import Visualizer
        return Visualizer._fig_to_base64(answer)
    return answer
```

## Phase 3: Add Specific Task Handlers (Days 4-5)

### 3.1 PDF Processing
```python
def handle_pdf_task(self, pdf_url, instructions):
    """Handle PDF extraction tasks"""
    from data_processor import DataProcessor
    
    # Download PDF
    pdf_content = DataProcessor.download_file(pdf_url)
    
    # Extract based on instructions
    if 'page' in instructions:
        page_num = self._extract_page_number(instructions)
        tables = DataProcessor.extract_pdf_tables(pdf_content, page_num)
    
    # Process table
    df = tables[0]
    
    # Perform operation (sum, average, etc.)
    if 'sum' in instructions.lower():
        column = self._extract_column_name(instructions)
        result = df[column].sum()
    
    return result
```

### 3.2 API Data Fetching
```python
def handle_api_task(self, api_url, instructions):
    """Handle API data fetching"""
    import requests
    
    # Parse headers from instructions if provided
    headers = self._extract_headers(instructions)
    
    response = requests.get(api_url, headers=headers)
    data = response.json()
    
    # Process based on instructions
    return self._process_json_data(data, instructions)
```

### 3.3 Visualization Tasks
```python
def handle_visualization_task(self, data, instructions):
    """Handle chart generation"""
    from visualization import Visualizer
    
    viz = Visualizer()
    
    if 'bar chart' in instructions.lower():
        return viz.create_bar_chart(data, x_col, y_col)
    elif 'line chart' in instructions.lower():
        return viz.create_line_chart(data, x_col, y_col)
    # Add more chart types...
```

## Phase 4: Error Handling and Retries (Day 6)

### 4.1 Add Retry Logic
```python
def solve_quiz_chain(self, initial_url):
    """Enhanced with retry logic"""
    max_retries = 3
    current_url = initial_url
    
    while current_url:
        for attempt in range(max_retries):
            try:
                result = self._solve_single_quiz(current_url)
                
                if result['correct']:
                    current_url = result.get('url')
                    break
                else:
                    logger.warning(f"Incorrect answer: {result.get('reason')}")
                    if attempt < max_retries - 1:
                        # Retry with refined approach
                        continue
                    else:
                        # Skip to next if available
                        current_url = result.get('url')
                        break
                        
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
```

## Phase 5: Testing with Demo (Day 7)

### 5.1 Test with Demo Endpoint
```bash
curl -X POST http://localhost:7860/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "23f1002487@ds.study.iitm.ac.in",
    "secret": "this-is-agni",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

### 5.2 Monitor Logs
```bash
tail -f app.log
```

## Phase 6: Deployment (Day 8)

### 6.1 Deploy to Hugging Face
```bash
# Initialize git if not done
git init
git add .
git commit -m "Initial commit"

# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/USERNAME/llm-quiz
git push hf main
```

### 6.2 Configure Secrets in HF
- Go to Space Settings → Variables
- Add: `OPENAI_API_KEY`, `STUDENT_EMAIL`, `STUDENT_SECRET`

### 6.3 Test Deployed Endpoint
```bash
curl -X POST https://USERNAME-llm-quiz.hf.space/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "23f1002487@ds.study.iitm.ac.in",
    "secret": "this-is-agni",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

## Phase 7: Fill Google Form (Day 9)

Fill the Google Form with:
1. Email: `23f1002487@ds.study.iitm.ac.in`
2. Secret: `this-is-agni`
3. System Prompt: See `prompts.md`
4. User Prompt: See `prompts.md`
5. API URL: Your HF Space URL
6. GitHub URL: Your public repo with MIT License

## Tips for Success

### Data Processing
- Always check file type before processing
- Handle missing data gracefully
- Use pandas for tabular data
- Use pdfplumber for PDFs

### LLM Usage
- Be specific in prompts
- Request structured JSON responses
- Parse responses carefully
- Handle LLM errors gracefully

### Time Management
- You have 3 minutes per quiz chain
- Optimize for speed but maintain accuracy
- Log everything for debugging
- Handle timeouts gracefully

### Common Pitfalls
- Not handling base64 encoding properly
- Forgetting to extract submit URL from instructions
- Not parsing LLM responses as JSON
- Hardcoding URLs instead of extracting from instructions
- Not handling retries properly

## Debugging Checklist

- [ ] Server starts without errors
- [ ] Health endpoint responds
- [ ] Authentication works (secret validation)
- [ ] Can fetch and render quiz pages
- [ ] LLM calls work
- [ ] Can download external files
- [ ] PDF extraction works
- [ ] Data processing works
- [ ] Answer submission works
- [ ] Handles incorrect answers
- [ ] Follows quiz chains
- [ ] Stays within 3-minute timeout
- [ ] Logs are informative

Good luck! 🚀
