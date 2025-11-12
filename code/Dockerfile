FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Playwright, tesseract-ocr, and other tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    git \
    # Playwright dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    # OCR support
    tesseract-ocr \
    tesseract-ocr-eng \
    # Image processing
    libopencv-dev \
    # Audio processing (optional)
    # ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only for smaller image)
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application code
COPY . .

# Set environment variables for Playwright and Python
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PYTHONUNBUFFERED=1

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run with uvicorn for FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "180"]
