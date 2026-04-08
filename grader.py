from actions import BiasAction, BiasLabel


class BiasGrader:
    """
    Computes deterministic rewards for agent actions.
    Returns scores between -1.0 and +1.0.
    """
    
    RELATED_BIASES = {
        BiasLabel.GENDER_BIAS: [BiasLabel.AGEISM],
        BiasLabel.RACIAL_BIAS: [BiasLabel.CULTURAL_BIAS],
        BiasLabel.CULTURAL_BIAS: [BiasLabel.RACIAL_BIAS],
        BiasLabel.POLITICAL_BIAS: [BiasLabel.COGNITIVE_BIAS],
        BiasLabel.COGNITIVE_BIAS: [BiasLabel.POLITICAL_BIAS],
        BiasLabel.AGEISM: [BiasLabel.GENDER_BIAS],
        BiasLabel.ABLEISM: [BiasLabel.GENDER_BIAS, BiasLabel.AGEISM],
        BiasLabel.NO_BIAS: [],
    }
    
    def compute_reward(self, action: BiasAction, ground_truth: dict) -> float:
        """
        Computes reward for agent's action.
        
        Scoring breakdown (total = 1.0 max):
        - LABEL ACCURACY: 0.4 points
        - SEVERITY ACCURACY: 0.2 points  
        - DEBIASING QUALITY: 0.4 points
        """
        reward = 0.0
        
        # LABEL ACCURACY (0.4 points)
        true_label = BiasLabel(ground_truth["true_label"])
        if action.label == true_label:
            reward += 0.4
        elif action.label in self.RELATED_BIASES.get(true_label, []):
            reward += 0.1
        elif true_label == BiasLabel.NO_BIAS and action.label != BiasLabel.NO_BIAS:
            # Said bias when no_bias: penalty
            reward -= 0.1
        elif true_label != BiasLabel.NO_BIAS and action.label == BiasLabel.NO_BIAS:
            # Said no_bias when bias exists: larger penalty
            reward -= 0.2
        # Otherwise completely wrong: 0.0
        
        # SEVERITY ACCURACY (0.2 points)
        true_severity = ground_truth["true_severity"]
        severity_diff = abs(action.severity - true_severity)
        if severity_diff <= 1:
            reward += 0.2
        elif severity_diff <= 3:
            reward += 0.1
        # More than 3 off: 0.0
        
        # DEBIASING QUALITY (0.4 points)
        original_text = ground_truth.get("text", "")
        debiased_version = ground_truth.get("debiased_version", "")
        
        if not action.corrected_text or action.corrected_text.strip() == "":
            reward -= 0.1  # Empty correction penalty
        elif action.corrected_text == original_text:
            # No attempt to debias
            pass  # 0.0 for this component
        else:
            # Compute similarity to ideal debiased version
            similarity = self.compute_similarity(action.corrected_text, debiased_version)
            reward += similarity * 0.4
        
        # Clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, reward))
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Simple word-overlap cosine similarity.
        Uses set operations for efficiency.
        """
        # Tokenize (simple split on whitespace)
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0
        
        intersection = tokens1 & tokens2
        
        # Cosine similarity approximation
        numerator = len(intersection)
        denominator = (len(tokens1) * len(tokens2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_feedback(self, reward: float, action: BiasAction, ground_truth: dict) -> str:
        """
        Returns human-readable feedback explaining the reward.
        """
        parts = []
        
        true_label = BiasLabel(ground_truth["true_label"])
        true_severity = ground_truth["true_severity"]
        
        # Label feedback
        if action.label == true_label:
            parts.append(f"✓ Correctly identified {action.label.value}.")
        elif action.label in self.RELATED_BIASES.get(true_label, []):
            parts.append(f"~ Related bias detected ({action.label.value}), but true label was {true_label.value}.")
        elif true_label == BiasLabel.NO_BIAS and action.label != BiasLabel.NO_BIAS:
            parts.append(f"✗ False positive: detected {action.label.value} when text was unbiased.")
        elif true_label != BiasLabel.NO_BIAS and action.label == BiasLabel.NO_BIAS:
            parts.append(f"✗ Missed {true_label.value} bias in the text.")
        else:
            parts.append(f"✗ Wrong bias type. Detected {action.label.value}, actual was {true_label.value}.")
        
        # Severity feedback
        severity_diff = abs(action.severity - true_severity)
        if severity_diff <= 1:
            parts.append(f"✓ Severity estimate accurate ({action.severity}/10 vs {true_severity}/10).")
        elif severity_diff <= 3:
            parts.append(f"~ Severity estimate close ({action.severity}/10 vs {true_severity}/10).")
        else:
            parts.append(f"✗ Severity estimate off ({action.severity}/10 vs {true_severity}/10).")
        
        # Correction feedback
        original_text = ground_truth.get("text", "")
        if not action.corrected_text or action.corrected_text.strip() == "":
            parts.append("✗ No debiased text provided.")
        elif action.corrected_text == original_text:
            parts.append("✗ Text was not modified (no debiasing attempt).")
        else:
            similarity = self.compute_similarity(action.corrected_text, ground_truth.get("debiased_version", ""))
            if similarity > 0.8:
                parts.append("✓ Excellent debiased version.")
            elif similarity > 0.5:
                parts.append("~ Acceptable debiased version, could be improved.")
            else:
                parts.append("✗ Debiased version differs significantly from expected.")
        
        parts.append(f"Reward: {reward:.2f}")
        
        return " ".join(parts)
