# Quiz Solver Loop Conditions and Logging

This document summarizes the key control-flow and logging behavior in the `QuizSolver` class, focusing on quiz-chain loops and retry logic.

## Quiz Chain Loop (`solve_quiz_chain`)

Method: `QuizSolver.solve_quiz_chain(initial_url: str)`

- **Initialization**
  - Tracks overall timing with `start_time = time.time()`.
  - `current_url` starts as `initial_url`.
  - `quiz_count` starts at 0.
  - Limits:
    - `max_quizzes = 20`
    - `max_retries_per_quiz = 3`

- **Main Loop Condition**
  - `while current_url and quiz_count < max_quizzes:`
    - Stops if there is no `current_url`.
    - Stops if `quiz_count` reaches 20.

- **Per-Quiz Timing and Timeout**
  - `quiz_start_time = time.time()` at the beginning of each quiz.
  - `elapsed = quiz_start_time - start_time` used for global timeout.
  - If `elapsed > 175` seconds:
    - Logs a warning: "Approaching 3-minute timeout limit, stopping".
    - Breaks out of the main loop.

- **Per-Quiz Retry Loop**
  - `quiz_solved = False` before the retry loop.
  - `for attempt in range(max_retries_per_quiz):`
    - Before each attempt, checks global time:
      - `current_elapsed = time.time() - start_time`
      - If `current_elapsed > 175`, logs "Time limit reached, cannot retry" and breaks.
    - Calls `await self.solve_single_quiz(current_url)`.

- **Result Handling**
  - Reads fields from the result JSON:
    - `is_correct = result.get('correct', False)`
    - `reason = result.get('reason', '')`
    - `next_url = result.get('url', '')`
  - If `is_correct` is `True`:
    - Logs success for that quiz.
    - Logs `reason` if provided.
    - Sets `current_url = next_url`.
    - Sets `quiz_solved = True` and breaks out of the retry loop.
  - If `is_correct` is `False`:
    - Logs a warning with the attempt number and reason.
    - Computes `time_since_quiz_start = time.time() - quiz_start_time`.
    - **Retry condition**:
      - If `time_since_quiz_start < 170` **and** `attempt < max_retries_per_quiz - 1`:
        - Logs "Retrying quiz (time since quiz start: ...s)...".
        - `await asyncio.sleep(1)` then continues to next attempt.
      - Otherwise:
        - If `next_url` is non-empty, logs and sets `current_url = next_url` to move on.
        - Breaks out of retry loop.

- **Exception Handling in Retry Loop**
  - If `solve_single_quiz` raises:
    - Logs the exception with attempt details.
    - Checks `current_elapsed = time.time() - start_time`.
    - If `current_elapsed > 175` or last attempt:
      - Logs "Cannot retry - time limit or max attempts reached" and returns.
    - Else, logs "Retrying after exception..." and `await asyncio.sleep(2)` before next attempt.

- **Post-Quiz Handling**
  - After retry loop, if `not quiz_solved and not current_url`:
    - Logs "No more quizzes to solve" and breaks main loop.
  - At the end, logs total quizzes solved and total time.

## Per-Quiz Flow (`solve_single_quiz`)

Method: `QuizSolver.solve_single_quiz(quiz_url: str)`

High-level steps and relevant conditions:

1. **Fetch Page**
   - `html_content = await self.fetch_quiz_page(quiz_url)`
   - `fetch_quiz_page` itself has a `@retry` decorator with:
     - `stop_after_attempt(3)`
     - `wait_exponential(multiplier=1, min=2, max=10)`
     - Retries on `httpx.HTTPError` and `asyncio.TimeoutError`.

2. **Parse Instructions**
   - `quiz_info = self.parse_quiz_instructions(html_content)`
   - Logging:
     - Whether `#result` div is used or full page text.
     - Number of elements with IDs and script tags.

3. **Understand Task via LLM**
   - `task_info = await self.understand_task_with_langchain(quiz_info)`
   - Logs task summary (e.g., meta-submission vs data analysis vs scraping).

4. **Download & Process Data**
   - `df, table_name = await self.process_data(task_info, quiz_url)`
   - Logging and conditions:
     - If `data_source_url` is missing:
       - Logs warning and returns `(None, None)`.
     - If `data_source_url` is relative (no `http://` or `https://`):
       - Uses `urljoin(quiz_url, data_url)` and logs the converted absolute URL.
     - After download, if content looks like HTML (`<` or `<!DOCTYPE` at start):
       - Logs warning and preview, skips data processing by returning `(None, None)`.

5. **Answer Computation**
   - If `df` and `table_name` are not `None`:
     - Logs "Data available - performing data analysis".
     - Calls `perform_analysis` (DuckDB + optional LLM backup).
   - Else:
     - Logs "No data available - answering question directly with LLM".
     - Calls `_answer_direct_question` with dynamic context.

6. **Formatting & Submission**
   - Formats answer based on `expected_answer_type` via `format_answer`.
   - Submits via `submit_answer`:
     - If `submit_url` is relative, converts to absolute using quiz URL and logs the conversion.
     - Logs payload URL and answer, and the JSON response from server.

## Logging Highlights

The `QuizSolver` uses `loguru` for structured logging with levels like INFO, SUCCESS, WARNING, and ERROR. Key logging points you may want to mirror:

- **Initialization**
  - `logger.info("QuizSolver initialized with model: {model_name}")`

- **Network Operations**
  - Fetching quiz pages, downloading data, and submitting answers all log:
    - Start of operation (INFO).
    - Success with sizes/URLs (SUCCESS/INFO).
    - Errors with exception details (ERROR).

- **Loop State**
  - For each quiz in the chain:
    - Logs quiz number, URL, and elapsed time.
  - For each attempt:
    - Logs whether the attempt was correct or incorrect.
    - Logs reasons for failure (from server JSON).
    - Logs retry decisions and timing.

- **Timeouts and Limits**
  - Global timeout (~3 minutes) enforced via `elapsed > 175` and `current_elapsed > 175` checks.
  - Per-quiz retry limit enforced via `max_retries_per_quiz = 3`.
  - Detailed warnings when these limits prevent further retries.

You can reuse these conditions and log messages as a reference pattern for any other long-running, multi-step workflows that need:

- Global time-budget control
- Per-item retry loops with backoff
- Clear logging for correctness, reasons, and transitions between steps.
