# Backend Dockerfile for 4-Color Theorem app
# Multi-stage build to keep the final image small

FROM python:3.11-slim AS base

# Install system dependencies for OpenCV, scipy, scikit-image, Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for runtime
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Drop privileges
RUN chown -R app:app /app
USER app

# Fly.io sets PORT env var; default to 8000 for local
ENV PORT=8000
EXPOSE 8000

# Use uvicorn directly so we can pass the port from the env
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
