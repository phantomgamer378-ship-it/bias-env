import os
import asyncio
from typing import Optional
from actions import BiasAction
from grader import BiasGrader


class BiasRubric:
    """
    LLM-based evaluation rubric for hackathon scoring.
    Falls back to deterministic grader if LLM API unavailable.
    """
    
    def __init__(self):
        self.api_base = os.environ.get("API_BASE_URL", "")
        self.model = os.environ.get("MODEL_NAME", "")
        self.token = os.environ.get("HF_TOKEN", "")
        self.grader = BiasGrader()
    
    async def evaluate(self, text: str, action: BiasAction, ground_truth: dict) -> dict:
        """
        Calls LLM API to evaluate agent's response quality.
        Falls back to deterministic grader if API unavailable.
        
        Returns:
            {
                "llm_score": float (0-10 average),
                "breakdown": dict,
                "feedback": str
            }
        """
        # Check if API is configured
        if not self.api_base:
            # Fallback to deterministic grader
            reward = self.grader.compute_reward(action, ground_truth)
            # Scale -1,1 to 0,10
            llm_score = (reward + 1.0) * 5.0
            return {
                "llm_score": llm_score,
                "breakdown": {
                    "bias_identification": llm_score,
                    "severity_rating": llm_score,
                    "correction_quality": llm_score,
                    "explanation_quality": llm_score,
                },
                "feedback": self.grader.get_feedback(reward, action, ground_truth)
            }
        
        # Try LLM evaluation
        try:
            llm_result = await self._call_llm_evaluator(text, action, ground_truth)
            return llm_result
        except Exception as e:
            # On failure, fallback to deterministic grader
            print(f"LLM evaluation failed: {e}")
            reward = self.grader.compute_reward(action, ground_truth)
            llm_score = (reward + 1.0) * 5.0
            return {
                "llm_score": llm_score,
                "breakdown": {
                    "bias_identification": llm_score,
                    "severity_rating": llm_score,
                    "correction_quality": llm_score,
                    "explanation_quality": llm_score,
                },
                "feedback": f"LLM eval failed, using deterministic grader. {self.grader.get_feedback(reward, action, ground_truth)}"
            }
    
    async def _call_llm_evaluator(self, text: str, action: BiasAction, ground_truth: dict) -> dict:
        """
        Makes actual LLM API call for evaluation.
        This is a placeholder that would call Hugging Face Inference API.
        """
        # This would be implemented with actual API calls
        # For now, fall back to deterministic scoring
        raise NotImplementedError("LLM API evaluation not yet implemented")
