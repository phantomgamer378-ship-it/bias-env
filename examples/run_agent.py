#!/usr/bin/env python3
"""
Example agents for BiasEnv.

Demonstrates:
1. Random agent - for testing environment mechanics
2. Rule-based agent - keyword matching baseline
"""

import asyncio
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import BiasEnv
from actions import BiasAction, BiasLabel


async def run_random_agent():
    """
    Runs a random agent for one full episode.
    Useful for testing the environment works correctly.
    """
    print("=" * 60)
    print("RANDOM AGENT")
    print("=" * 60)
    
    env = BiasEnv()
    
    # Connect and reset
    obs = await env.reset()
    print(f"✅ Reset OK - Episode started")
    print(f"   First text: {obs.text[:60]}...\n")
    
    step = 0
    while not obs.done:
        step += 1
        
        # Random action
        label = random.choice(list(BiasLabel))
        severity = random.randint(0, 10)
        
        action = BiasAction(
            label=label,
            severity=severity,
            corrected_text=f"[Random correction of: {obs.text[:30]}...]",
            explanation="Random agent - no reasoning"
        )
        
        print(f"Step {step}: {obs.text[:50]}...")
        print(f"  → Action: {action.label.value}, Severity: {action.severity}")
        
        obs = await env.step(action)
        
        print(f"  ← Reward: {obs.reward:+.2f} | Cumulative: {obs.cumulative_reward:.2f}")
        print(f"  ← Feedback: {obs.feedback[:80]}...\n")
    
    # Episode summary
    summary = env.get_episode_summary()
    print("=" * 60)
    print("EPISODE SUMMARY (Random Agent)")
    print("=" * 60)
    print(f"Total Reward: {summary['total_reward']:.2f}")
    print(f"Steps: {summary['steps_completed']}")
    print(f"Average Reward: {summary['avg_reward']:.2f}")
    print("=" * 60)
    
    return summary['total_reward']


async def run_rule_based_agent():
    """
    Simple rule-based agent that uses keyword matching.
    
    Keywords like 'he/she', 'old people', 'those people' 
    trigger specific bias labels.
    
    Shows baseline performance before LLM training.
    """
    print("\n" + "=" * 60)
    print("RULE-BASED AGENT")
    print("=" * 60)
    
    # Define keyword patterns
    keywords = {
        BiasLabel.GENDER_BIAS: [
            "he", "she", "man", "woman", "male", "female", "his", "her",
            "salesman", "saleswoman", "businessman", "nurse", "female doctor",
            "men are", "women are"
        ],
        BiasLabel.RACIAL_BIAS: [
            "those people", "that neighborhood", "minority", "immigrants",
            "urban youth", "asian", "black", "white people", "taking our jobs",
            "refusing to integrate"
        ],
        BiasLabel.CULTURAL_BIAS: [
            "western", "civilized", "traditional dress", "unprofessional",
            "native english", "developing world"
        ],
        BiasLabel.POLITICAL_BIAS: [
            "only naive", "real americans", "biased media", "elites",
            "out of touch", "radical"
        ],
        BiasLabel.COGNITIVE_BIAS: [
            "always do it this way", "first candidate", "heard one story",
            "availability", "anchoring"
        ],
        BiasLabel.AGEISM: [
            "young people", "old people", "millennial", "boomer",
            "elderly", "fresh young", "stale old"
        ],
        BiasLabel.ABLEISM: [
            "disabled", "normal people", "suffers from", "poor thing",
            "can't handle", "handicapped"
        ]
    }
    
    def classify_text(text: str) -> tuple:
        """Classify text based on keyword matching."""
        text_lower = text.lower()
        
        for label, words in keywords.items():
            for word in words:
                if word in text_lower:
                    return label, 6, f"Detected keyword: '{word}'"
        
        return BiasLabel.NO_BIAS, 0, "No bias keywords detected"
    
    def generate_correction(text: str, label: BiasLabel) -> str:
        """Generate a simple correction based on label."""
        corrections = {
            BiasLabel.GENDER_BIAS: text.replace("salesman", "salesperson").replace("He ", "They ").replace("She ", "They ").replace(" his ", " their ").replace(" her ", " their "),
            BiasLabel.RACIAL_BIAS: text.replace("those people", "people in that community").replace("urban youth", "young people in the city"),
            BiasLabel.AGEISM: text.replace("young people today", "people today").replace("old people", "older adults"),
            BiasLabel.ABLEISM: text.replace("suffers from", "lives with").replace("normal people", "people without disabilities"),
        }
        return corrections.get(label, text)
    
    env = BiasEnv()
    obs = await env.reset()
    
    print(f"✅ Reset OK - Episode started\n")
    
    step = 0
    while not obs.done:
        step += 1
        text = obs.text
        
        # Classify
        label, severity, explanation = classify_text(text)
        
        # Generate correction
        corrected = generate_correction(text, label)
        
        action = BiasAction(
            label=label,
            severity=severity,
            corrected_text=corrected,
            explanation=explanation
        )
        
        print(f"Step {step}: {text[:50]}...")
        print(f"  → Action: {action.label.value}, Severity: {action.severity}")
        print(f"  → Explanation: {explanation}")
        
        obs = await env.step(action)
        
        print(f"  ← Reward: {obs.reward:+.2f} | Cumulative: {obs.cumulative_reward:.2f}")
        print(f"  ← Feedback: {obs.feedback[:60]}...\n")
    
    # Episode summary
    summary = env.get_episode_summary()
    print("=" * 60)
    print("EPISODE SUMMARY (Rule-Based Agent)")
    print("=" * 60)
    print(f"Total Reward: {summary['total_reward']:.2f}")
    print(f"Steps: {summary['steps_completed']}")
    print(f"Average Reward: {summary['avg_reward']:.2f}")
    print("=" * 60)
    
    return summary['total_reward']


async def compare_agents():
    """Run both agents and compare performance."""
    random_score = await run_random_agent()
    rule_score = await run_rule_based_agent()
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Random Agent:     {random_score:+.2f}")
    print(f"Rule-Based Agent: {rule_score:+.2f}")
    print(f"Improvement:      {rule_score - random_score:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run example agents on BiasEnv")
    parser.add_argument(
        "--agent",
        choices=["random", "rule", "compare"],
        default="compare",
        help="Which agent to run"
    )
    
    args = parser.parse_args()
    
    if args.agent == "random":
        asyncio.run(run_random_agent())
    elif args.agent == "rule":
        asyncio.run(run_rule_based_agent())
    else:
        asyncio.run(compare_agents())
