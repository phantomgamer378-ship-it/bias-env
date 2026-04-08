"""
BiasEnv - Reinforcement Learning Environment for Bias Detection
OpenEnv-compatible implementation
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from actions import BiasAction, BiasLabel
from grader import BiasGrader
from dataset import BiasDataset
from rubric import BiasRubric


class BiasObservation(BaseModel):
    """
    What the agent sees each step.
    """
    text: str = Field(..., description="The text to analyze")
    step_number: int = Field(..., description="Current step in episode")
    total_steps: int = Field(..., description="Max steps per episode (10)")
    reward: float = Field(..., description="Reward from last action (0.0 on reset)")
    cumulative_reward: float = Field(..., description="Total reward so far this episode")
    feedback: str = Field(..., description="Human-readable feedback on last action")
    done: bool = Field(..., description="Is episode over?")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class EnvironmentMetadata:
    """Metadata about the environment."""
    def __init__(self, name: str, description: str, version: str, author: str):
        self.name = name
        self.description = description
        self.version = version
        self.author = author


class BiasEnv:
    """
    An RL environment for training AI agents to detect and correct bias in text.
    
    Agents receive a text snippet and must classify the bias type, 
    rate severity, and provide a corrected debiased version.
    Reward is computed via deterministic grading + optional LLM evaluation.
    """
    
    metadata = EnvironmentMetadata(
        name="BiasEnv",
        description="""
            An RL environment for training AI agents to detect and 
            correct bias in text. Agents receive a text snippet and 
            must classify the bias type, rate severity, and provide 
            a corrected debiased version. Reward is computed via 
            deterministic grading + optional LLM evaluation.
        """,
        version="1.0.0",
        author="BiasEnv Team"
    )
    
    def __init__(self):
        self.dataset = BiasDataset()
        self.grader = BiasGrader()
        self.rubric = BiasRubric()
        self.max_steps = 10  # 10 examples per episode
        self._reset_state()
    
    def _reset_state(self):
        """Reset internal state for a new episode."""
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.current_example = None
        self.last_reward = 0.0
        self.last_feedback = "Episode started. Analyze the text below."
        self.episode_history = []
    
    async def reset(self) -> BiasObservation:
        """
        Start fresh episode. Pick first random example from dataset.
        Returns initial observation.
        """
        self._reset_state()
        self.current_example = self.dataset.get_random()
        
        return BiasObservation(
            text=self.current_example["text"],
            step_number=0,
            total_steps=self.max_steps,
            reward=0.0,
            cumulative_reward=0.0,
            feedback=self.last_feedback,
            done=False,
            info={}
        )
    
    async def step(self, action: BiasAction) -> BiasObservation:
        """
        Process agent's action for current text.
        
        1. Compute reward via grader
        2. Get feedback string
        3. Increment step counter
        4. Load next example if not done
        5. Check if episode complete (step >= max_steps)
        6. Return new observation
        """
        # Compute reward for the action
        self.last_reward = self.grader.compute_reward(action, self.current_example)
        self.cumulative_reward += self.last_reward
        
        # Get feedback
        self.last_feedback = self.grader.get_feedback(
            self.last_reward, action, self.current_example
        )
        
        # Store step in history
        self.episode_history.append({
            "step": self.current_step,
            "text": self.current_example["text"],
            "action": action.dict(),
            "reward": self.last_reward,
            "true_label": self.current_example["true_label"].value,
        })
        
        # Increment step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Prepare info dict
        info = {}
        if done:
            # Reveal true labels when episode is complete
            info = {
                "true_label": self.current_example["true_label"].value,
                "true_severity": self.current_example["true_severity"],
                "debiased_version": self.current_example["debiased_version"],
                "explanation": self.current_example["explanation"],
                "episode_summary": {
                    "total_reward": round(self.cumulative_reward, 2),
                    "steps": self.current_step,
                    "avg_reward": round(self.cumulative_reward / self.current_step, 2) if self.current_step > 0 else 0
                }
            }
        
        # Load next example if not done
        if not done:
            self.current_example = self.dataset.get_random()
            next_text = self.current_example["text"]
        else:
            next_text = "Episode complete."
        
        return BiasObservation(
            text=next_text,
            step_number=self.current_step,
            total_steps=self.max_steps,
            reward=self.last_reward,
            cumulative_reward=self.cumulative_reward,
            feedback=self.last_feedback,
            done=done,
            info=info
        )
    
    async def state(self) -> BiasObservation:
        """
        Returns current observation without advancing the environment.
        If reset() has not been called, return a default observation.
        """
        if self.current_example is None:
            return BiasObservation(
                text="Call reset() to start an episode.",
                step_number=0,
                total_steps=self.max_steps,
                reward=0.0,
                cumulative_reward=0.0,
                feedback="Environment not initialized.",
                done=False,
                info={}
            )
        
        return BiasObservation(
            text=self.current_example["text"],
            step_number=self.current_step,
            total_steps=self.max_steps,
            reward=self.last_reward,
            cumulative_reward=self.cumulative_reward,
            feedback=self.last_feedback,
            done=self.current_step >= self.max_steps,
            info={}
        )
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of completed episode."""
        return {
            "total_reward": round(self.cumulative_reward, 2),
            "steps_completed": self.current_step,
            "avg_reward": round(self.cumulative_reward / self.current_step, 2) if self.current_step > 0 else 0,
            "history": self.episode_history
        }
