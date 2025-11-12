"""
LLM Analysis Quiz API Server - PRODUCTION READY
FastAPI application with proper background tasks, validation, and error handling
"""
import os
import sys
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, HttpUrl, Field, validator
from pydantic_settings import BaseSettings
from loguru import logger
import httpx

from ultimate_quiz_solver import UltimateQuizSolver


# ============================================
# CONFIGURATION WITH VALIDATION
# ============================================

class Settings(BaseSettings):
    """Application settings with validation"""
    student_email: EmailStr = Field(..., description="Student email address")
    student_secret: str = Field(..., min_length=1, description="Authentication secret")
    openai_api_key: str = Field(..., min_length=20, description="OpenAI API key")
    port: int = Field(default=7860, ge=1, le=65535)
    max_concurrent_quizzes: int = Field(default=3, ge=1, le=10)
    
    class Config:
        env_file = ".env"
        env_prefix = ""
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if v.startswith('your_') or v == '':
            raise ValueError("OpenAI API key must be set to a valid key")
        return v


# Load and validate settings at startup
try:
    settings = Settings()
    logger.info("✓ Configuration loaded successfully")
except Exception as e:
    logger.error(f"✗ Configuration error: {e}")
    logger.error("Please set required environment variables: STUDENT_EMAIL, STUDENT_SECRET, OPENAI_API_KEY")
    sys.exit(1)


# ============================================
# CONFIGURE LOGGING
# ============================================

# Ensure logs directory exists early (HF Spaces fresh container)
import os as _os
_os.makedirs("logs", exist_ok=True)

# Remove default logger
logger.remove()

# Add console logger with colors
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

# Add file logger with rotation
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    enqueue=True  # Thread-safe
)

# Add error-only file
logger.add(
    "logs/errors_{time:YYYY-MM-DD}.log",
    rotation="100 MB",
    retention="30 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
    enqueue=True
)

logger.info("✓ Logging configured (logs directory ensured)")


# ============================================
# RATE LIMITING
# ============================================

from collections import defaultdict
from time import time

class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        now = time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > minute_ago]
        
        # Check limit
        if len(self.requests[key]) >= self.requests_per_minute:
            return False
        
        # Add new request
        self.requests[key].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=10)


# ============================================
# QUIZ SOLVER POOL
# ============================================

class QuizSolverPool:
    """Manage concurrent quiz solvers with limit"""
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Try to acquire a slot"""
        async with self.lock:
            if self.active_count >= self.max_concurrent:
                return False
            self.active_count += 1
            return True
    
    async def release(self):
        """Release a slot"""
        async with self.lock:
            self.active_count -= 1

solver_pool = QuizSolverPool(max_concurrent=settings.max_concurrent_quizzes)


# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Application starting up")
    
    # Verify OpenAI API key works
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                timeout=10.0
            )
            if response.status_code == 200:
                logger.success("✓ OpenAI API key validated")
            else:
                logger.warning(f"⚠ OpenAI API key validation returned: {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Failed to validate OpenAI API key: {e}")
    
    # Verify Playwright installation
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
        logger.success("✓ Playwright browser verified")
    except Exception as e:
        logger.error(f"✗ Playwright not properly installed: {e}")
        logger.error("Run: playwright install chromium")
    
    yield
    
    # Shutdown
    logger.info("🛑 Application shutting down")
    logger.info(f"Active quiz solvers: {solver_pool.active_count}")


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="LLM Analysis Quiz API",
    description="Production-ready API for solving data analysis quizzes",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# PYDANTIC MODELS
# ============================================

class QuizRequest(BaseModel):
    email: EmailStr = Field(..., description="Student email address")
    secret: str = Field(..., min_length=1, description="Secret authentication string")
    url: HttpUrl = Field(..., description="Quiz URL to solve")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL is safe"""
        url_str = str(v)
        # Block dangerous schemes
        if url_str.startswith(('file://', 'javascript:', 'data:')):
            raise ValueError("Invalid URL scheme")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "23f1002487@ds.study.iitm.ac.in",
                "secret": "this-is-agni",
                "url": "https://example.com/quiz-123"
            }
        }


class QuizResponse(BaseModel):
    status: str
    message: str
    url: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: dict


# ============================================
# BACKGROUND TASK
# ============================================

async def solve_quiz_background(quiz_url: str, request_id: str):
    """Background task to solve quiz with proper error handling"""
    logger.info(f"[{request_id}] Starting quiz solving: {quiz_url}")
    
    try:
        # Create solver instance for this request
        async with UltimateQuizSolver(
            email=settings.student_email,
            secret=settings.student_secret,
            openai_api_key=settings.openai_api_key
        ) as solver:
            # Solve quiz chain with timeout protection
            await asyncio.wait_for(
                solver.solve_quiz_chain(quiz_url),
                timeout=175.0  # 175 seconds = 2:55 (5 second buffer)
            )
            logger.success(f"[{request_id}] ✓ Quiz chain completed")
    
    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] ✗ Quiz solving timed out (175s)")
    except Exception as e:
        logger.exception(f"[{request_id}] ✗ Quiz solving failed: {e}")
    finally:
        # Release solver slot
        await solver_pool.release()
        logger.info(f"[{request_id}] Released solver slot, active: {solver_pool.active_count}")


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Analysis Quiz API",
        "version": "3.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /quiz": "Submit quiz task",
            "GET /health": "Health check",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Deep health check with component validation"""
    components = {
        "api": "healthy",
        "config": "healthy",
        "rate_limiter": "healthy",
        "solver_pool": f"{solver_pool.active_count}/{solver_pool.max_concurrent}"
    }
    
    # Check OpenAI API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                timeout=5.0
            )
            components["openai_api"] = "healthy" if response.status_code == 200 else "degraded"
    except:
        components["openai_api"] = "unhealthy"
    
    # Overall status
    overall_status = "healthy" if all(
        v in ["healthy", "operational"] or "/" in str(v) 
        for v in components.values()
    ) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "components": components
    }


@app.post("/quiz", response_model=QuizResponse, tags=["Quiz"], 
          responses={
              200: {"description": "Quiz accepted for processing"},
              403: {"description": "Invalid credentials"},
              422: {"description": "Validation error"},
              429: {"description": "Rate limit exceeded"},
              503: {"description": "Service busy"}
          })
async def handle_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint to receive quiz tasks
    
    Uses FastAPI BackgroundTasks for reliable processing:
    - Non-daemon threads (won't be killed on shutdown)
    - Proper error tracking
    - Resource management
    - Concurrent request limiting
    """
    # Generate request ID for tracking
    request_id = f"quiz-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{id(request)}"
    
    logger.info(f"[{request_id}] Quiz request received: {request.url}")
    logger.info(f"[{request_id}] Email: {request.email}")
    
    try:
        # Verify credentials
        if request.secret != settings.student_secret:
            logger.warning(f"[{request_id}] Invalid secret")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid secret"
            )
        
        if request.email != settings.student_email:
            logger.warning(f"[{request_id}] Invalid email: {request.email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid email"
            )
        
        # Rate limiting
        client_key = f"{request.email}:{request.secret}"
        if not rate_limiter.is_allowed(client_key):
            logger.warning(f"[{request_id}] Rate limit exceeded")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Max 10 requests per minute."
            )
        
        # Check solver pool capacity
        if not await solver_pool.acquire():
            logger.warning(f"[{request_id}] Service busy, max concurrent quizzes reached")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service busy. Maximum {settings.max_concurrent_quizzes} concurrent quizzes allowed."
            )
        
        quiz_url = str(request.url)
        
        # Add to background tasks (PROPER WAY - not daemon threads!)
        background_tasks.add_task(solve_quiz_background, quiz_url, request_id)
        
        logger.success(f"[{request_id}] Quiz task accepted and queued")
        
        return QuizResponse(
            status="accepted",
            message="Quiz processing started",
            url=quiz_url,
            request_id=request_id
        )
        
    except HTTPException:
        # Release slot if we acquired it
        if solver_pool.active_count > 0:
            await solver_pool.release()
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Error handling quiz: {e}")
        if solver_pool.active_count > 0:
            await solver_pool.release()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """Handle validation errors with detailed logging"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": str(exc)
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )


# ============================================
# STARTUP
# ============================================

if __name__ == '__main__':
    import uvicorn
    
    logger.info(f"Starting server on port {settings.port}")
    logger.info(f"Student email: {settings.student_email}")
    logger.info(f"Max concurrent quizzes: {settings.max_concurrent_quizzes}")
    
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=settings.port,
        log_config=None  # Use loguru instead
    )
