#!/usr/bin/env python3
"""Inference script for BiasEnv with validator output format."""

import os
import asyncio
import json
import sys
from typing import Optional

import httpx


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def build_prompt(text: str, feedback: str) -> str:
    """Build the LLM prompt for bias detection."""
    return f"""You are an expert bias detection agent. Analyze this text and respond with ONLY valid JSON matching this schema exactly:
{{
  "label": one of [gender_bias, racial_bias, cultural_bias, political_bias, cognitive_bias, ageism, ableism, no_bias],
  "severity": integer 0-10,
  "corrected_text": "your debiased rewrite here",
  "explanation": "why this text is biased"
}}

Text to analyze: {text}
Last feedback: {feedback}

Respond with valid JSON only:"""


def rule_based_response(prompt: str) -> str:
    """Simple rule-based agent as fallback."""
    text_start = prompt.find("Text to analyze: ")
    if text_start == -1:
        return '{"label": "no_bias", "severity": 0, "corrected_text": "", "explanation": "parse error"}'
    
    text_start += len("Text to analyze: ")
    text_end = prompt.find("Last feedback:", text_start)
    if text_end == -1:
        text = prompt[text_start:].strip()
    else:
        text = prompt[text_start:text_end].strip()
    
    text_lower = text.lower()
    
    # Keyword-based detection
    gender_keywords = ["he", "she", "man", "woman", "male", "female", "his", "her", "salesman", "nurse"]
    racial_keywords = ["those people", "minority", "immigrant", "urban youth", "asian", "black", "white"]
    age_keywords = ["young", "old", "millennial", "boomer", "elderly", "fresh"]
    
    if any(kw in text_lower for kw in gender_keywords):
        corrected = text.replace("salesman", "salesperson").replace("He ", "They ").replace("She ", "They ")
        return json.dumps({"label": "gender_bias", "severity": 6, "corrected_text": corrected, "explanation": "Gender-coded language detected."})
    elif any(kw in text_lower for kw in racial_keywords):
        return json.dumps({"label": "racial_bias", "severity": 7, "corrected_text": text, "explanation": "Racially coded language detected."})
    elif any(kw in text_lower for kw in age_keywords):
        return json.dumps({"label": "ageism", "severity": 5, "corrected_text": text, "explanation": "Age-related assumptions detected."})
    
    return json.dumps({"label": "no_bias", "severity": 0, "corrected_text": text, "explanation": "No clear bias indicators found."})


async def call_llm(prompt: str) -> str:
    """Call LLM API with fallback to rule-based."""
    if not API_BASE_URL or "localhost" in API_BASE_URL:
        return rule_based_response(prompt)
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.1}}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(API_BASE_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            return str(result)
        except Exception as e:
            return rule_based_response(prompt)


def parse_llm_response(response_text: str, original_text: str) -> dict:
    """Parse LLM response into action dict."""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")
        
        json_str = response_text[start:end+1]
        data = json.loads(json_str)
        
        return {
            "label": data.get("label", "no_bias"),
            "severity": max(0, min(10, int(data.get("severity", 0)))),
            "corrected_text": data.get("corrected_text", original_text),
            "explanation": data.get("explanation", "parse error")
        }
    except Exception as e:
        return {
            "label": "no_bias",
            "severity": 0,
            "corrected_text": original_text,
            "explanation": "parse error"
        }


async def run_inference(base_url: str = "http://localhost:7860") -> float:
    """Run one full episode via HTTP API with validator output format."""
    async with httpx.AsyncClient() as client:
        try:
            # Reset to start episode
            reset_resp = await client.post(f"{base_url}/reset", timeout=10.0)
            obs = reset_resp.json()
            
            # Print START marker
            print("[START] task=BiasEnv", flush=True)
            
            step_count = 0
            total_reward = 0.0
            
            while not obs.get('done', False):
                step_count += 1
                text = obs.get('text', '')
                
                # Build prompt and call LLM
                prompt = build_prompt(text, obs.get('feedback', ''))
                llm_response = await call_llm(prompt)
                
                # Parse response into action
                action = parse_llm_response(llm_response, text)
                
                # Take step via API
                step_resp = await client.post(f"{base_url}/step", json=action, timeout=10.0)
                obs = step_resp.json()
                
                reward = obs.get('reward', 0)
                total_reward += reward
                
                # Print STEP marker
                print(f"[STEP] step={step_count} reward={reward:.2f}", flush=True)
            
            # Episode complete - print END marker
            avg_reward = total_reward / step_count if step_count > 0 else 0
            print(f"[END] task=BiasEnv score={avg_reward:.2f} steps={step_count}", flush=True)
            
            return avg_reward
            
        except Exception as e:
            print(f"[END] task=BiasEnv score=0.00 steps=0", flush=True)
            return 0.0


if __name__ == "__main__":
    base_url = "http://localhost:7860"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    try:
        final_reward = asyncio.run(run_inference(base_url))
        sys.exit(0)
    except Exception as e:
        print(f"[END] task=BiasEnv score=0.00 steps=0", flush=True)
        sys.exit(0)
