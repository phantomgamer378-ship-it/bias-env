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

# Start the OpenEnv HTTP server
# Note: Using Python directly since openenv CLI may not be available
CMD ["python3", "-c", "
import asyncio
import uvicorn
from fastapi import FastAPI
from environment import BiasEnv
from actions import BiasAction

app = FastAPI()
env = BiasEnv()

@app.get('/health')
def health():
    return {'status': 'healthy', 'env': 'BiasEnv'}

@app.post('/reset')
async def reset():
    obs = await env.reset()
    return obs.dict()

@app.post('/step')
async def step(action: BiasAction):
    obs = await env.step(action)
    return obs.dict()

@app.get('/state')
async def state():
    obs = await env.state()
    return obs.dict()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)
"]
