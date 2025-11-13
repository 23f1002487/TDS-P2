"""
LLM Analysis Quiz API Server
FastAPI application with proper background tasks, validation, and error handling
"""
import os
import sys
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, EmailStr, HttpUrl, Field, validator
from pydantic_settings import BaseSettings
from loguru import logger
import httpx
import json

from quiz_solver import QuizSolver


# region Configuration
# ============================================
# CONFIGURATION WITH VALIDATION
# ============================================

class Settings(BaseSettings):
    """Application settings with validation"""
    student_email: EmailStr = Field(..., description="Student email address")
    student_secret: str = Field(..., min_length=1, description="Authentication secret")
    aipipe_token: str = Field(..., min_length=20, description="AIPipe API token")
    aipipe_base_url: str = Field(default="https://aipipe.org/openai/v1", description="AIPipe base URL")
    openai_model: str = Field(default="gpt-4o-mini", description="Model name without provider prefix")
    port: int = Field(default=7860, ge=1, le=65535)
    
    @validator('openai_model')
    def remove_provider_prefix(cls, v):
        """Remove provider prefix like 'openai/' if present"""
        if '/' in v:
            return v.split('/')[-1]
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = ""
    
    @validator('aipipe_token')
    def validate_aipipe_token(cls, v):
        if v.startswith('your-') or v == '':
            raise ValueError("AIPipe token must be set to a valid token")
        return v


# Load and validate settings at startup
try:
    settings = Settings()
    logger.info("✓ Configuration loaded successfully")
    logger.info(f"Using AIPipe Base URL: {settings.aipipe_base_url}")
    logger.info(f"Using Model: {settings.openai_model}")
except Exception as e:
    logger.error(f"✗ Configuration error: {e}")
    logger.error("Please set required environment variables: STUDENT_EMAIL, STUDENT_SECRET, AIPIPE_TOKEN")
    sys.exit(1)

# endregion

# region Logging
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

# Add file logger with rotation (Developement or local development Only)
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

# endregion

# region Lifespan Management
# ============================================
# LIFESPAN MANAGEMENT
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Application starting up")
    
    # Log AIPipe configuration (validation happens during actual API calls)
    logger.info(f"✓ AIPipe configured: {settings.aipipe_base_url}")
    logger.info(f"✓ Model: {settings.openai_model}")
    logger.info(f"✓ Token length: {len(settings.aipipe_token)} chars")
    
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

# endregion

# region FastAPI App
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to catch invalid JSON early
@app.middleware("http")
async def catch_json_errors(request: Request, call_next):
    """Catch JSON decode errors and return 400"""
    if request.method in ["POST", "PUT", "PATCH"] and "application/json" in request.headers.get("content-type", ""):
        try:
            # Try to read and parse the body
            body = await request.body()
            if body:
                json.loads(body)
            # Create a new request with the body since it was consumed
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON received on {request.url.path}: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid JSON",
                    "detail": f"Failed to decode JSON: {str(e)}",
                    "message": "Request body must be valid JSON"
                }
            )
        except Exception:
            pass  # Let other errors be handled normally
    
    response = await call_next(request)
    return response

# endregion

# region Pydantic Models
# ============================================
# PYDANTIC MODELS
# ============================================

class QuizRequest(BaseModel):
    email: EmailStr = Field(..., description="Student email address")
    secret: str = Field(..., min_length=1, description="Secret authentication string")
    url: HttpUrl = Field(..., description="Quiz URL to solve")
    
    class Config:
        extra = "ignore"  # Ignore unknown fields instead of raising validation error
        json_schema_extra = {
            "example": {
                "email": "my_id@ds.study.iitm.ac.in",
                "secret": "this-is-secret",
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

# endregion

# region Background Task
# ============================================
# BACKGROUND TASK
# ============================================

async def solve_quiz_background(quiz_url: str, request_id: str):
    """Background task to solve quiz with proper error handling"""
    logger.info(f"[{request_id}] Starting quiz solving: {quiz_url}")
    
    try:
        # Create solver instance for this request
        async with QuizSolver(
            email=settings.student_email,
            secret=settings.student_secret,
            aipipe_token=settings.aipipe_token,
            aipipe_base_url=settings.aipipe_base_url,
            model_name=settings.openai_model
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
        logger.info(f"[{request_id}] Quiz solving completed")

# endregion

# region API Endpoints
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
        "config": "healthy"
    }
    
    # Check AIPipe configuration (actual validation happens during API calls)
    components["aipipe_api"] = "operational" if settings.aipipe_token and len(settings.aipipe_token) > 10 else "not_configured"
    
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
              400: {"description": "Invalid JSON format"},
              403: {"description": "Invalid credentials"},
              422: {"description": "Validation error"}
          })
async def handle_quiz(request: QuizRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint to receive quiz tasks
    
    Uses FastAPI BackgroundTasks for reliable processing:
    - Non-daemon threads (won't be killed on shutdown)
    - Proper error tracking
    - Resource management
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
        
        quiz_url = str(request.url)
        
        # Add to background tasks
        background_tasks.add_task(solve_quiz_background, quiz_url, request_id)
        
        logger.success(f"[{request_id}] Quiz task accepted and queued")
        
        return QuizResponse(
            status="accepted",
            message="Quiz processing started",
            url=quiz_url,
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Error handling quiz: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# endregion

# region Error Handlers
# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors (422)"""
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
            "body": str(exc.body) if hasattr(exc, 'body') else None
        }
    )


@app.exception_handler(json.JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    """Handle invalid JSON (400)"""
    logger.warning(f"Invalid JSON on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid JSON",
            "detail": f"Failed to decode JSON: {str(exc)}",
            "message": "Request body must be valid JSON"
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

# endregion

# region Startup
# ============================================
# STARTUP
# ============================================

if __name__ == '__main__':
    import uvicorn
    
    logger.info(f"Starting server on port {settings.port}")
    logger.info(f"Student email: {settings.student_email}")
    
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=settings.port,
        log_config=None  # Use loguru instead
    )

# endregion
