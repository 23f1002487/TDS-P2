# Mock Quiz Server - Usage Guide

## 🎨 Mock Makeup Quiz Server

A local FastAPI server with 15 makeup quiz questions for testing your quiz solver app.

## 🚀 Quick Start

### 1. Start the Server
```bash
python3 mock_quiz_server.py
```

The server will run on `http://localhost:8000`

### 2. Available Endpoints

#### Web Interface (HTML)
- **Home Page**: http://localhost:8000/
- **Demo Quiz** (3 questions): http://localhost:8000/demo
- **Full Quiz** (15 questions): http://localhost:8000/full-quiz
- **Custom Quiz**: http://localhost:8000/random-quiz/5 (replace 5 with any number 1-15)
- **API Docs** (Interactive): http://localhost:8000/docs

#### API Endpoints (JSON)
- **Health Check**: `GET /health`
- **All Questions**: `GET /api/questions`

### 3. Test Your App with the Mock Server

Update your `test_live_app.py` or test script to use:
```python
QUIZ_URL = "http://localhost:8000/demo"
```

Then run your tests:
```bash
python3 test_live_app.py
```

### 4. Test the Mock Server

Run the verification script:
```bash
python3 test_mock_server.py
```

## 📝 Sample Quiz Questions

The server includes 15 makeup-related questions covering topics like:
- Makeup Basics (primer, foundation)
- Contouring & Highlighting
- Eye Makeup Techniques
- Color Correction
- Makeup Tools
- Advanced Techniques (baking, cut crease)
- Lip Products
- Makeup Chemistry

## 🔧 Integration with Your App

### NEW: Submit Endpoint for Answer Validation

The mock server now includes a `/submit` endpoint that receives and validates quiz answers!

**Endpoint**: `POST /submit`

**Request Format**:
```json
{
  "email": "student@example.com",
  "quiz_url": "http://localhost:8000/demo",
  "answers": [
    {"question_id": 1, "answer": 1},
    {"question_id": 2, "answer": 2}
  ]
}
```

**Response Format**:
```json
{
  "status": "success",
  "message": "Quiz submitted successfully",
  "results": {
    "total_questions": 2,
    "correct_answers": 2,
    "score_percentage": 100.0,
    "grade": "A (Excellent)",
    "timestamp": "2025-11-12T21:30:00"
  },
  "detailed_results": [...]
}
```

### Example: Complete Flow Test

```bash
# Run the complete integration test
python3 test_complete_flow.py
```

### Example: Testing with curl
```bash
# Get health status
curl http://localhost:8000/health

# Get all questions as JSON
curl http://localhost:8000/api/questions

# Submit answers for validation
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "quiz_url": "http://localhost:8000/demo",
    "answers": [
      {"question_id": 1, "answer": 1},
      {"question_id": 2, "answer": 2}
    ]
  }'

# Get submission history
curl http://localhost:8000/submissions
```

### Example: Testing with Python requests
```python
import requests

# Get quiz questions
response = requests.get("http://localhost:8000/api/questions")
questions = response.json()

# Submit answers
submission = {
    "email": "test@example.com",
    "quiz_url": "http://localhost:8000/demo",
    "answers": [
        {"question_id": 1, "answer": 1},
        {"question_id": 2, "answer": 2}
    ]
}
response = requests.post("http://localhost:8000/submit", json=submission)
results = response.json()
print(f"Score: {results['results']['score_percentage']}%")
```

## 🎯 Use Cases

1. **Local Testing**: Test your quiz solver without relying on external services
2. **Development**: Develop and debug your app with consistent quiz data
3. **CI/CD**: Use in automated testing pipelines
4. **Demos**: Show how your app works without external dependencies

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal where the server is running, or:
```bash
pkill -f mock_quiz_server.py
```

## 📊 Server Information

- **Port**: 8000
- **Host**: 0.0.0.0 (accessible from network)
- **Framework**: FastAPI
- **Total Questions**: 15
- **Question Format**: Multiple choice with 4 options each
- **CORS**: Enabled for all origins (suitable for local testing)

## 💡 Tips

- The server shuffles questions for `/demo` and `/random-quiz/` endpoints
- All questions have correct answers marked
- HTML pages include interactive forms
- API responses are in JSON format
- The server logs all requests for debugging

## 🔗 Next Steps

1. ✅ Server is running at http://localhost:8000
2. Test with your quiz solver app
3. Use `/api/questions` to see all available questions
4. Check `/docs` for interactive API documentation

Enjoy testing your app! 🚀
