#!/usr/bin/env python3
import sys
# CRITICAL: Print START marker immediately before any imports that might fail
print("[START] task=BiasEnv", flush=True)

try:
    import httpx
except ImportError:
    print("[STEP] step=1 reward=0.00", flush=True)
    print("[END] task=BiasEnv score=0.00 steps=1", flush=True)
    sys.exit(0)

async def run():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{base}/reset", timeout=10)
            obs = r.json()
            step = 0
            total = 0.0
            while not obs.get("done", False):
                step += 1
                text = obs.get("text", "")
                action = {"label": "no_bias", "severity": 0, "corrected_text": text, "explanation": "test"}
                r = await c.post(f"{base}/step", json=action, timeout=10)
                obs = r.json()
                reward = obs.get("reward", 0)
                total += reward
                print(f"[STEP] step={step} reward={reward:.2f}", flush=True)
            avg = total / step if step > 0 else 0
            print(f"[END] task=BiasEnv score={avg:.2f} steps={step}", flush=True)
    except:
        print("[STEP] step=1 reward=0.00", flush=True)
        print("[END] task=BiasEnv score=0.00 steps=1", flush=True)

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(run())
    except:
        print("[STEP] step=1 reward=0.00", flush=True)
        print("[END] task=BiasEnv score=0.00 steps=1", flush=True)
    sys.exit(0)
