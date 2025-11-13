# LLM Content and Rate Limiting Analysis

## Summary of Changes Made

### 1. Rate Limit Protection Added
- ✅ Added `RateLimitError` and `APIError` imports from OpenAI
- ✅ Added retry decorators with exponential backoff to all LLM-calling methods:
  - `understand_task_with_langchain()`: 5 retries, 4-60 sec backoff
  - `_llm_assisted_analysis()`: 5 retries, 4-60 sec backoff  
  - `_create_visualization()`: 5 retries, 4-60 sec backoff
- ✅ Added specific error handling for `RateLimitError` with warning logs
- ✅ Retry strategy: exponential backoff with multiplier=2, min=4s, max=60s

### 2. Content Size Monitoring & Limiting

#### Method 1: `understand_task_with_langchain()`
**What's sent to LLM:**
- Quiz instructions text (from HTML page)
- Available links as JSON
- System prompt with JSON schema

**Size limits implemented:**
- ✅ Max quiz text: 8,000 chars (~2,000 tokens)
- ✅ Logging: Quiz text length, links JSON length, total prompt length
- ✅ Estimated tokens: length/4 (rough approximation)

**Typical content:**
```
Quiz text: 200-5000 chars
Links JSON: 50-500 chars
System prompt: ~800 chars
Total: 1,050-6,300 chars (~260-1,575 tokens)
Max tokens requested: 1,000
```

#### Method 2: `_llm_assisted_analysis()`
**What's sent to LLM:**
- Task summary
- Operation type
- DataFrame columns, dtypes, shape
- Sample data (first 10 rows as string)

**Size limits implemented:**
- ✅ Max sample data: 4,000 chars (~1,000 tokens)
- ✅ Logging: Sample size, DataFrame shape, total prompt length
- ✅ Uses `df.head(10).to_string()` - limited to 10 rows

**Typical content:**
```
Task info: 100-500 chars
Data metadata: 100-300 chars
Sample data: 500-4,000 chars (10 rows max)
System prompt: ~500 chars
Total: 1,200-5,300 chars (~300-1,325 tokens)
Max tokens requested: 1,000
```

#### Method 3: `_create_visualization()`
**What's sent to LLM:**
- Task summary
- Column names only (not data)
- Chart type options

**Size limits:**
- ✅ Logging: Total prompt length
- ✅ Very small prompts (~200-500 chars)

**Typical content:**
```
Task summary: 50-200 chars
Columns list: 20-200 chars
System prompt: ~300 chars
Total: 370-700 chars (~90-175 tokens)
Max tokens requested: 500
```

## Total Token Usage Per Quiz

**Estimated per quiz attempt:**
- Task understanding: 260-1,575 input + 1,000 output = ~1,835 tokens max
- Data analysis: 300-1,325 input + 1,000 output = ~2,325 tokens max
- Visualization: 90-175 input + 500 output = ~675 tokens max

**Total per quiz: ~4,835 tokens max**

With retries (5 attempts max each):
- Worst case: 4,835 × 5 = ~24,175 tokens per quiz

## Rate Limit Error Analysis from test3.log

**Error encountered:**
```
Error code: 429 - {'error': {'message': 'You exceeded your current quota, 
please check your plan and billing details.', 
'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}
```

**Location:** `understand_task_with_langchain()` on first LLM call
**Page size:** 528 bytes (small, not the issue)
**Root cause:** API quota/billing limit exceeded, NOT content size

## Retry Strategy Details

```python
@retry(
    stop=stop_after_attempt(5),           # Try 5 times
    wait=wait_exponential(multiplier=2, min=4, max=60),  # 4s, 8s, 16s, 32s, 60s
    retry=retry_if_exception_type((RateLimitError, APIError, asyncio.TimeoutError)),
    reraise=True
)
```

**Backoff timing:**
- Attempt 1: immediate
- Attempt 2: wait 4 seconds
- Attempt 3: wait 8 seconds
- Attempt 4: wait 16 seconds
- Attempt 5: wait 32 seconds
- Total: up to 60 seconds of waiting

## Logging Enhancements

All LLM calls now log:
1. Input data sizes before processing
2. Prompt length in chars and estimated tokens
3. Rate limit warnings with retry indication
4. API errors with full context

## Recommendations

1. **Check API Quota:** The error is a billing/quota issue, not content size
2. **Monitor Logs:** New logging will show exact sizes being sent
3. **Content is Safe:** All content is limited and well within token limits
4. **Retry Logic:** Should handle temporary rate limits automatically
5. **Consider:** Adding a global rate limiter to prevent hitting API limits

## Test with New Changes

Run the app again and check logs for:
- "Quiz text length: X chars"
- "Sending prompt to LLM (total length: X chars, ~X tokens)"
- "Rate limit error - will retry with exponential backoff"
- Data sample truncation warnings if content is too large
