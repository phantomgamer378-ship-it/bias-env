#!/usr/bin/env python3
"""
Inference script for BiasEnv - MANDATORY for hackathon submission.

Runs an LLM agent through one full episode of the BiasEnv environment.
"""

import os
import asyncio
import json
from typing import Optional

import httpx
from actions import BiasAction, BiasLabel
from environment import BiasEnv


# Configuration from environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
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


def parse_llm_response(response_text: str, original_text: str) -> BiasAction:
    """
    Parse LLM response into BiasAction.
    Handles JSON parse errors gracefully.
    """
    try:
        # Try to find JSON in the response
        # Look for content between curly braces
        start = response_text.find("{")
        end = response_text.rfind("}")
        
        if start == -1 or end == -1:
            raise ValueError("No JSON found in response")
        
        json_str = response_text[start:end+1]
        data = json.loads(json_str)
        
        # Validate label
        label_str = data.get("label", "no_bias")
        try:
            label = BiasLabel(label_str)
        except ValueError:
            label = BiasLabel.NO_BIAS
        
        # Validate severity
        severity = data.get("severity", 0)
        if not isinstance(severity, int) or severity < 0 or severity > 10:
            severity = 0
        
        return BiasAction(
            label=label,
            severity=severity,
            corrected_text=data.get("corrected_text", original_text),
            explanation=data.get("explanation", "parse error")
        )
    
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return BiasAction(
            label=BiasLabel.NO_BIAS,
            severity=0,
            corrected_text=original_text,
            explanation="parse error"
        )
    except Exception as e:
        print(f"Parse error: {e}")
        return BiasAction(
            label=BiasLabel.NO_BIAS,
            severity=0,
            corrected_text=original_text,
            explanation="parse error"
        )


async def call_llm(prompt: str) -> str:
    """
    Call LLM API with the prompt.
    
    If API_BASE_URL is not set, uses a simple rule-based fallback.
    """
    if not API_BASE_URL:
        # Fallback: rule-based agent for testing
        return rule_based_response(prompt)
    
    # Call actual LLM API
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "return_full_text": False
        }
    }
    
    if MODEL_NAME:
        payload["model"] = MODEL_NAME
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_BASE_URL}",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            else:
                return str(result)
                
        except Exception as e:
            print(f"LLM API error: {e}")
            return rule_based_response(prompt)


def rule_based_response(prompt: str) -> str:
    """
    Simple rule-based agent as fallback when no LLM API available.
    Uses keyword matching to detect bias.
    """
    # Extract text from prompt
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
    ableist_keywords = ["disabled", "normal", "suffers from", "handicapped"]
    
    detected_label = BiasLabel.NO_BIAS
    severity = 0
    explanation = "No bias detected based on keyword analysis."
    corrected = text
    
    # Check for gender bias
    if any(kw in text_lower for kw in gender_keywords):
        detected_label = BiasLabel.GENDER_BIAS
        severity = 6
        explanation = "Detected gender-coded language or assumptions."
        corrected = text.replace("salesman", "salesperson").replace("He ", "They ").replace("She ", "They ")
    
    # Check for racial/ethnic bias
    elif any(kw in text_lower for kw in racial_keywords):
        detected_label = BiasLabel.RACIAL_BIAS
        severity = 7
        explanation = "Detected potentially racially coded language."
        corrected = text.replace("those people", "people from that community")
    
    # Check for ageism
    elif any(kw in text_lower for kw in age_keywords):
        detected_label = BiasLabel.AGEISM
        severity = 5
        explanation = "Detected age-related assumptions."
        corrected = text.replace("old", "experienced").replace("young", "early-career")
    
    # Check for ableism
    elif any(kw in text_lower for kw in ableist_keywords):
        detected_label = BiasLabel.ABLEISM
        severity = 6
        explanation = "Detected ableist language."
        corrected = text.replace("suffers from", "lives with").replace("normal people", "most people")
    
    # If no keywords match, could be unbiased text
    if detected_label == BiasLabel.NO_BIAS:
        return json.dumps({
            "label": "no_bias",
            "severity": 0,
            "corrected_text": text,
            "explanation": "No clear bias indicators found in text."
        })
    
    return json.dumps({
        "label": detected_label.value,
        "severity": severity,
        "corrected_text": corrected,
        "explanation": explanation
    })


async def run_inference(base_url: str = "http://localhost:7860") -> float:
    """
    Run one full episode using the LLM as agent.
    
    Args:
        base_url: URL of the running BiasEnv environment
        
    Returns:
        Final cumulative reward
    """
    print(f"Connecting to BiasEnv at {base_url}...")
    
    # Create environment instance
    env = BiasEnv()
    
    # Reset to start episode
    obs = await env.reset()
    print(f"✅ Episode started. Analyzing {obs.total_steps} text snippets...\n")
    
    step_count = 0
    
    while not obs.done:
        step_count += 1
        print(f"--- Step {step_count}/{obs.total_steps} ---")
        print(f"Text: {obs.text[:80]}...")
        
        # Build prompt for LLM
        prompt = build_prompt(obs.text, obs.feedback)
        
        # Call LLM
        llm_response = await call_llm(prompt)
        
        # Parse response into action
        action = parse_llm_response(llm_response, obs.text)
        
        print(f"Action: {action.label.value}, Severity: {action.severity}")
        print(f"Correction: {action.corrected_text[:60]}...")
        
        # Take step in environment
        obs = await env.step(action)
        
        print(f"Reward: {obs.reward:.2f}")
        print(f"Feedback: {obs.feedback[:100]}...\n")
    
    # Episode complete
    summary = env.get_episode_summary()
    
    print("=" * 50)
    print("EPISODE COMPLETE")
    print("=" * 50)
    print(f"Total Reward: {summary['total_reward']:.2f}")
    print(f"Steps Completed: {summary['steps_completed']}")
    print(f"Average Reward per Step: {summary['avg_reward']:.2f}")
    print("=" * 50)
    
    return summary['total_reward']


if __name__ == "__main__":
    # Allow base URL override from command line
    import sys
    
    base_url = "http://localhost:7860"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    final_reward = asyncio.run(run_inference(base_url))
    
    # Exit with reward as exit code for automation (capped to 0-255)
    exit_code = int((final_reward + 1.0) * 127.5)
    exit_code = max(0, min(255, exit_code))
    exit(exit_code)
