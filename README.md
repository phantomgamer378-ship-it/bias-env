---
title: BiasEnv
emoji: 🔍
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---

# BiasEnv — Bias Detection RL Environment

## What This Environment Does

BiasEnv is a reinforcement learning environment where AI agents learn to detect and correct biased text. Agents receive text snippets and must classify the bias type, rate severity (0-10), and provide a debiased rewrite. The environment scores actions using deterministic grading and optional LLM evaluation.

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `label` | BiasLabel enum | Type of bias detected (gender_bias, racial_bias, cultural_bias, political_bias, cognitive_bias, ageism, ableism, no_bias) |
| `severity` | int (0-10) | Agent's confidence in bias level |
| `corrected_text` | str | Agent's proposed debiased rewrite |
| `explanation` | str | Why the agent thinks this text is biased |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | The text snippet to analyze |
| `step_number` | int | Current step in episode (0-9) |
| `total_steps` | int | Max steps per episode (10) |
| `reward` | float | Reward from last action |
| `cumulative_reward` | float | Total reward so far |
| `feedback` | str | Human-readable feedback on last action |
| `done` | bool | Is episode complete? |
| `info` | dict | Extra metadata (reveals true labels when done) |

## Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| Label Accuracy | 0.4 | Exact match +0.4, related +0.1, wrong 0.0, false positive -0.1, missed -0.2 |
| Severity Accuracy | 0.2 | Within 1 point +0.2, within 3 +0.1, off 0.0 |
| Debiasing Quality | 0.4 | Cosine similarity to ideal debiased version |

## Reward Range

- **Minimum**: -1.0
- **Maximum**: +1.0

## Episode Structure

- Steps per episode: 10
- Each step: new text snippet to analyze
- Episode ends after 10 steps
- True labels revealed in `info` dict when done

## Quick Start

```python
import asyncio
from environment import BiasEnv
from actions import BiasAction, BiasLabel

async def main():
    env = BiasEnv()
    
    # Start episode
    obs = await env.reset()
    print(f"Text to analyze: {obs.text}")
    
    # Take action
    action = BiasAction(
        label=BiasLabel.GENDER_BIAS,
        severity=7,
        corrected_text="Engineers are skilled problem-solvers.",
        explanation="Uses masculine pronouns exclusively."
    )
    
    obs = await env.step(action)
    print(f"Reward: {obs.reward}")
    print(f"Feedback: {obs.feedback}")
    
    # Continue until done...

asyncio.run(main())
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | No | LLM API endpoint for rubric evaluation |
| `MODEL_NAME` | No | Model identifier for LLM evaluation |
| `HF_TOKEN` | No | Hugging Face token for API access |

## Bias Categories Supported

1. **gender_bias** — Gender-coded language, pronoun assumptions, stereotypical traits
2. **racial_bias** — Racially coded language, ethnic stereotypes, immigration framing
3. **cultural_bias** — Western-centric assumptions, cultural superiority claims
4. **political_bias** — Loaded framing, dismissive language, false dichotomies
5. **cognitive_bias** — Status quo bias, anchoring, availability heuristic
6. **ageism** — Generational stereotypes, age-based assumptions
7. **ableism** — Disability framing, "normal" as default, pity language
8. **no_bias** — Neutral text without identifiable bias

## Evaluation Criteria

### Deterministic Grader (always runs)
- Exact label match: +0.4
- Related bias category: +0.1
- Severity within 1 point: +0.2
- Debiasing similarity: up to +0.4
- False positives/negatives: -0.1 to -0.2

### LLM Rubric (optional)
When `API_BASE_URL` is set, an LLM provides additional evaluation:
- Bias identification quality (0-10)
- Severity appropriateness (0-10)
- Correction quality (0-10)
- Explanation accuracy (0-10)

## Files

| File | Purpose |
|------|---------|
| `environment.py` | Core RL environment with reset(), step(), state() |
| `actions.py` | BiasAction and BiasLabel definitions |
| `dataset.py` | 35+ text examples with ground truth |
| `grader.py` | Deterministic reward computation |
| `rubric.py` | LLM-based evaluation |
| `inference.py` | LLM agent script (hackathon submission) |
| `examples/run_agent.py` | Example agents for testing |

## Local Development

```bash
# Create virtual environment
python -m venv biasenv
source biasenv/bin/activate  # or biasenv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -c "from environment import BiasEnv; print('✅ Import OK')"

# Run inference
python inference.py

# Build Docker image
docker build -t bias-env .
docker run -p 7860:7860 bias-env
```

## API Endpoints

When running via Docker:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Take action (accepts BiasAction JSON) |
| `/state` | GET | Get current observation |

## Hackathon Submission

This environment was built for the Meta PyTorch OpenEnv Hackathon.

- **Framework**: OpenEnv-compatible RL environment
- **Language**: Python 3.11
- **Deployment**: Hugging Face Spaces (Docker)
- **Inference**: `inference.py` (required file)

## License

Apache 2.0 — See LICENSE file for details.
