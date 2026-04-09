#!/usr/bin/env python3
import sys
print("[START] task=BiasEnv", flush=True)

try:
    import os
    import asyncio
    import httpx
    import json
except ImportError:
    print("[STEP] step=1 reward=0.00", flush=True)
    print("[END] task=BiasEnv score=0.00 steps=1", flush=True)
    sys.exit(0)

# Use LiteLLM proxy from hackathon
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")

def build_prompt(text, feedback):
    return f"""Analyze this text for bias. Respond with ONLY JSON:
{{"label": "gender_bias|racial_bias|cultural_bias|political_bias|cognitive_bias|ageism|ableism|no_bias", "severity": 0-10, "corrected_text": "...", "explanation": "..."}}

Text: {text}
Feedback: {feedback}

JSON:"""

async def call_llm(prompt):
    """Call LLM through LiteLLM proxy."""
    if not API_BASE_URL:
        # Fallback if no proxy configured
        return None
    
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return None

def parse_response(text):
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON")
        data = json.loads(text[start:end+1])
        return {
            "label": data.get("label", "no_bias"),
            "severity": max(0, min(10, int(data.get("severity", 0)))),
            "corrected_text": data.get("corrected_text", ""),
            "explanation": data.get("explanation", "")
        }
    except:
        return {"label": "no_bias", "severity": 0, "corrected_text": "", "explanation": "parse error"}

async def run():
    # Environment URL (local or remote)
    env_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    
    try:
        async with httpx.AsyncClient() as c:
            # Reset environment
            r = await c.post(f"{env_url}/reset", timeout=10)
            obs = r.json()
            
            step = 0
            total = 0.0
            
            while not obs.get("done", False):
                step += 1
                text = obs.get("text", "")
                feedback = obs.get("feedback", "")
                
                # Build prompt and call LLM through proxy
                prompt = build_prompt(text, feedback)
                llm_response = await call_llm(prompt)
                
                if llm_response:
                    action = parse_response(llm_response)
                else:
                    # Fallback action if LLM call fails
                    action = {"label": "no_bias", "severity": 0, "corrected_text": text, "explanation": "fallback"}
                
                # Take step
                r = await c.post(f"{env_url}/step", json=action, timeout=10)
                obs = r.json()
                
                reward = obs.get("reward", 0)
                total += reward
                print(f"[STEP] step={step} reward={reward:.2f}", flush=True)
            
            avg = total / step if step > 0 else 0
            print(f"[END] task=BiasEnv score={avg:.2f} steps={step}", flush=True)
            
    except Exception as e:
        print("[STEP] step=1 reward=0.00", flush=True)
        print("[END] task=BiasEnv score=0.00 steps=1", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except:
        print("[STEP] step=1 reward=0.00", flush=True)
        print("[END] task=BiasEnv score=0.00 steps=1", flush=True)
    sys.exit(0)
