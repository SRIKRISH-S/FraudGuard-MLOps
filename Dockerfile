FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKDIR=/app
ENV PYTHONPATH=/app

WORKDIR $WORKDIR

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs tests api dashboard src

# Copy project files
COPY api/ api/
COPY dashboard/ dashboard/
COPY src/ src/
COPY tests/ tests/
COPY models/ models/

# Expose ports: 8000 for FastAPI, 8501 for Streamlit, 5000 for MLflow
EXPOSE 8000 10000 8501 5000

# Default command binding to Render's dynamic PORT (defaults to 8000 locally)
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
