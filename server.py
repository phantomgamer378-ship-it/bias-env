#!/usr/bin/env python3
"""
FastAPI server for BiasEnv on Hugging Face Spaces.
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from environment import BiasEnv
from actions import BiasAction

app = FastAPI(title="BiasEnv", description="RL Environment for Bias Detection")
env = BiasEnv()


class ActionRequest(BaseModel):
    label: str
    severity: int
    corrected_text: str
    explanation: str


@app.get("/")
def root():
    return {
        "name": "BiasEnv",
        "version": "1.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "env": "BiasEnv"}


@app.post("/reset")
async def reset():
    obs = await env.reset()
    return obs.dict()


@app.post("/step")
async def step(action: ActionRequest):
    bias_action = BiasAction(
        label=action.label,
        severity=action.severity,
        corrected_text=action.corrected_text,
        explanation=action.explanation
    )
    obs = await env.step(bias_action)
    return obs.dict()


@app.get("/state")
async def state():
    obs = await env.state()
    return obs.dict()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
