from pydantic import BaseModel, Field
from enum import Enum


class BiasLabel(str, Enum):
    GENDER_BIAS = "gender_bias"
    RACIAL_BIAS = "racial_bias"
    CULTURAL_BIAS = "cultural_bias"
    POLITICAL_BIAS = "political_bias"
    COGNITIVE_BIAS = "cognitive_bias"
    AGEISM = "ageism"
    ABLEISM = "ableism"
    NO_BIAS = "no_bias"


class BiasAction(BaseModel):
    """
    Agent action for bias detection.
    
    The agent must provide:
    1. What type of bias is present (or no_bias)
    2. Severity score 0-10
    3. A corrected/debiased version of the text
    4. Explanation of the bias found
    """
    label: BiasLabel = Field(..., description="Type of bias detected")
    severity: int = Field(..., ge=0, le=10, description="Severity score 0-10")
    corrected_text: str = Field(..., description="Agent's proposed debiased rewrite")
    explanation: str = Field(..., description="Why agent thinks this is biased")
