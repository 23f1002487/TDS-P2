# Test Directory

This directory contains the mock quiz server and test scripts for your quiz solver application.

## Files

### Mock Quiz Server
- **`mock_quiz_server.py`** - FastAPI server with 15 makeup quiz questions
  - Provides quiz endpoints (demo, full, random)
  - Validates and scores submitted answers
  - Tracks submission history
  - Run with: `python3 mock_quiz_server.py`
  - Server runs on: `http://localhost:8000`

### Test Scripts
- **`test_live_app.py`** - Main test script for your quiz solver app
  - Tests all API endpoints (root, health, quiz)
  - Validates request/response formats
  - Tests quiz submission to HF Space or local server
  - Run with: `python3 test_live_app.py`

- **`test_complete_flow.py`** - End-to-end integration test
  - Tests mock server health
  - Gets quiz questions
  - Submits answers for validation
  - Checks scoring and submission history
  - Run with: `python3 test_complete_flow.py`

- **`test_mock_server.py`** - Mock server verification
  - Verifies mock server is running
  - Shows available endpoints
  - Displays sample questions
  - Run with: `python3 test_mock_server.py`

### Documentation
- **`mock_server_guide.md`** - Complete guide for using the mock quiz server
  - API documentation
  - Usage examples
  - Integration instructions

## Quick Start

1. **Start the mock quiz server:**
   ```bash
   python3 mock_quiz_server.py
   ```

2. **In another terminal, run tests:**
   ```bash
   # Test mock server
   python3 test_mock_server.py
   
   # Test complete flow
   python3 test_complete_flow.py
   
   # Test your app (HF Space or local)
   python3 test_live_app.py
   ```

## Server Endpoints

The mock server provides:
- `GET /` - Home page
- `GET /demo` - Demo quiz (3 questions)
- `GET /full-quiz` - Full quiz (15 questions)
- `GET /api/questions` - All questions as JSON
- `POST /submit` - Submit answers for validation
- `GET /submissions` - View submission history
- `GET /health` - Server health check
- `GET /docs` - Interactive API docs

## Testing Your App

Update `test_live_app.py` configuration:
```python
BASE_URL = "https://your-app.hf.space"  # Your app server
QUIZ_URL = "http://localhost:8000/demo"  # Mock quiz server
EMAIL = "your-email@ds.study.iitm.ac.in"
SECRET = "your-secret"
```

Then run: `python3 test_live_app.py`
