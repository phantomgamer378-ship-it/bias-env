FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces runs on port 7860 — mandatory
EXPOSE 7860

# Environment variables (values injected by HF Spaces secrets)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# Install uv and dependencies
RUN pip install uv
RUN uv pip install --system -e .

# Start the FastAPI server
CMD ["server"]
