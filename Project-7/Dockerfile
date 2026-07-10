# ============================================================
# Agentic AI Travel Planning Assistant — Dockerfile
# ============================================================
FROM python:3.12-slim

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY . .

# Create directories that may not be in the repo
RUN mkdir -p database logs

# Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command — override with `docker run ... python -m pytest` for tests
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
