# TDS Project 2 - Quiz Solver API

Production-ready FastAPI application for solving data analysis quizzes using LLMs via AIPipe.

## 🚀 Live Deployment

- **Hugging Face Space**: https://huggingface.co/spaces/vSaketh/TDS-P2
- **API Documentation**: https://vsaketh-tds-p2.hf.space/docs
- **Health Check**: https://vsaketh-tds-p2.hf.space/health

## ✅ Test Results

All endpoints tested and operational:
- ✅ Root endpoint (GET /)
- ✅ Health check (GET /health)
- ✅ API documentation (GET /docs)
- ✅ Input validation
- ✅ Rate limiting

## 📋 API Endpoints

### 1. Root Endpoint
```bash
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```bash
GET /health
```
Returns system health status and component information.

### 3. Submit Quiz (Main Endpoint)
```bash
POST /quiz
Content-Type: application/json

{
  "email": "your-email@ds.study.iitm.ac.in",
  "secret": "your-secret-string",
  "url": "https://quiz-url.com"
}
```

**Response (202 Accepted):**
```json
{
  "status": "accepted",
  "message": "Quiz task queued for processing",
  "url": "https://quiz-url.com",
  "request_id": "unique-request-id"
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Update credentials in test_live_app.py first
python3 test_live_app.py
```

### Manual Testing with cURL

**1. Check health:**
```bash
curl https://vsaketh-tds-p2.hf.space/health
```

**2. Submit a quiz:**
```bash
curl -X POST https://vsaketh-tds-p2.hf.space/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@ds.study.iitm.ac.in",
    "secret": "your-secret",
    "url": "https://quiz-url.com"
  }'
```

**3. View interactive docs:**
Visit: https://vsaketh-tds-p2.hf.space/docs

## 🔧 Configuration

The application uses these environment variables (set in HF Space Secrets):

```bash
STUDENT_EMAIL=your-email@ds.study.iitm.ac.in
STUDENT_SECRET=your-secret-string
AIPIPE_TOKEN=your-aipipe-token
AIPIPE_BASE_URL=https://aipipe.org/openai/v1
OPENAI_MODEL=openai/gpt-4o-mini
PORT=7860
LOG_LEVEL=INFO
MAX_CONCURRENT_QUIZZES=3
```

## 🏗️ Architecture

### Tech Stack
- **FastAPI**: High-performance async web framework
- **OpenAI (via AIPipe)**: LLM inference for quiz understanding
- **Playwright**: Headless browser for dynamic content
- **DuckDB**: In-memory SQL database for data processing
- **Pandas/Polars**: Data manipulation
- **Matplotlib/Plotly**: Data visualization

### Features
- ✅ Async/await throughout for high performance
- ✅ Background task processing (non-blocking)
- ✅ Rate limiting and concurrency control
- ✅ Comprehensive input validation
- ✅ Structured logging (loguru)
- ✅ Health checks and monitoring
- ✅ OpenAPI/Swagger documentation
- ✅ CORS enabled for web integration

### Processing Pipeline
1. **Fetch Quiz**: Use Playwright to load dynamic content
2. **Parse Instructions**: Extract task requirements
3. **Understand Task**: Use LLM to analyze what's needed
4. **Download Data**: Fetch required datasets
5. **Process Data**: Load into DuckDB, clean and transform
6. **Perform Analysis**: Execute SQL queries or complex operations
7. **Generate Answer**: Format result (number/string/visualization)
8. **Submit Answer**: POST to submission endpoint

## 📊 Monitoring

### Health Status
The `/health` endpoint provides detailed component status:

```json
{
  "status": "healthy",
  "components": {
    "api": "healthy",
    "config": "healthy",
    "rate_limiter": "healthy",
    "solver_pool": "0/3",
    "aipipe_api": "configured"
  }
}
```

### Logs
Logs are available in the HF Space logs viewer or in the `logs/` directory:
- `app_YYYY-MM-DD.log` - All application logs
- `errors_YYYY-MM-DD.log` - Error-only logs

## 🚦 Rate Limiting

- Maximum 3 concurrent quiz processing tasks
- Additional requests queued or rejected (429 Too Many Requests)
- Configurable via `MAX_CONCURRENT_QUIZZES` environment variable

## 🔒 Security Features

- Email validation (must be valid EmailStr)
- URL scheme validation (only http/https allowed)
- Secret string validation
- Input sanitization
- CORS configuration
- No dangerous file:// or javascript: URLs

## 📦 Dependencies

See `requirements.txt` for full list. Key dependencies:
- fastapi==0.104.1
- openai>=1.6.1,<2.0.0
- playwright==1.40.0
- duckdb==0.9.2
- pandas==2.1.4
- httpx==0.25.2

## 🐛 Troubleshooting

### Issue: 422 Validation Error
**Solution**: Check that email, secret, and URL are properly formatted

### Issue: 429 Too Many Requests
**Solution**: Wait for previous tasks to complete or increase MAX_CONCURRENT_QUIZZES

### Issue: Quiz processing times out
**Solution**: Default timeout is 175 seconds. Complex tasks may need optimization

## 📝 Development

### Local Setup
```bash
# Clone repository
git clone https://github.com/23f1002487/TDS-P2.git
cd TDS-P2

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Set environment variables
cp .env.example .env
# Edit .env with your credentials

# Run locally
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Running Tests
```bash
# Run test suite
python3 test_live_app.py

# Run with pytest (if implemented)
pytest test/
```

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

This is a project for TDS (Tools in Data Science) course at IIT Madras.

## 📞 Support

- GitHub Issues: https://github.com/23f1002487/TDS-P2/issues
- HF Space Discussions: https://huggingface.co/spaces/vSaketh/TDS-P2/discussions
