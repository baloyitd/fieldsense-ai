"""
FieldSense AI v3.1 - Agent-Assisted Calibration Reasoning

Uses MiroThinker agent to reason about high-entropy plays during LoRA calibration.
Provides refined loss weights for focused fine-tuning on team-specific patterns.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from .load import MiroThinkerAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationReasoner:
    """
    Agent-assisted reasoning for LoRA calibration.

    Analyzes high-entropy plays to determine:
    - Zone priorities (which field areas to focus on)
    - Team bias patterns (attack tendencies)
    - Refined loss weights for LoRA targets
    """

    def __init__(self, agent: Optional[MiroThinkerAgent] = None):
        """
        Initialize calibration reasoner.

        Args:
            agent: MiroThinker agent instance (creates new if None)
        """
        self.agent = agent or MiroThinkerAgent(offline_mode=True)

    def reason_calibration(
        self,
        plays_json: List[Dict[str, Any]],
        entropy_threshold: float = 0.8
    ) -> Dict[str, float]:
        """
        Reason about calibration priorities for high-entropy plays.

        Args:
            plays_json: List of play data with predictions and entropy
            entropy_threshold: Minimum entropy to trigger reasoning

        Returns:
            Dictionary of zone weights {zone_name: weight}
        """
        # Filter high-entropy plays
        high_entropy_plays = [
            p for p in plays_json
            if p.get('entropy', 0.0) > entropy_threshold
        ]

        if len(high_entropy_plays) == 0:
            logger.info("No high-entropy plays, using default weights")
            return self._get_default_weights()

        logger.info(f"Reasoning about {len(high_entropy_plays)} high-entropy plays...")

        # Build reasoning prompt
        prompt = self._build_calibration_prompt(high_entropy_plays)

        # Generate reasoning
        if self.agent.enabled and self.agent.model is not None:
            output = self._generate_with_agent(prompt)
        else:
            logger.info("Agent not enabled, using simulation")
            output = self._simulate_reasoning(high_entropy_plays)

        # Parse weights from output
        weights = self._parse_weights(output)

        logger.info(f"Agent refined weights: {weights}")

        return weights

    def _build_calibration_prompt(self, plays: List[Dict[str, Any]]) -> str:
        """Build prompt for calibration reasoning."""
        # Analyze play patterns
        zones = []
        entropies = []
        outcomes = []

        for play in plays:
            if 'target_zone' in play:
                zones.append(play['target_zone'])
            if 'entropy' in play:
                entropies.append(play['entropy'])
            if 'outcome' in play:
                outcomes.append(play['outcome'])

        # Summarize patterns
        zone_counts = {}
        for zone in zones:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

        avg_entropy = np.mean(entropies) if entropies else 0.0

        # Build prompt
        prompt = f"""Given calibration plays with high uncertainty:

Total high-entropy plays: {len(plays)}
Average entropy: {avg_entropy:.3f}
Zone distribution: {zone_counts}

Reason about calibration priorities step-by-step:

Step 1: Analyze Entropy Patterns
- Identify which zones have highest uncertainty
- Determine if uncertainty is systematic (team bias) or random

Step 2: Detect Team Bias
- Look for consistent attack patterns (e.g., left-side preference)
- Identify under-represented zones in training
- Find strategic patterns that need emphasis

Step 3: Compute Priority Weights
- Assign higher weights to zones with systematic patterns
- Balance coverage across field regions
- Focus on team-specific tendencies

Output format (JSON):
{{
  "left_attack": <weight 0.0-2.0>,
  "right_attack": <weight 0.0-2.0>,
  "center_attack": <weight 0.0-2.0>,
  "left_defense": <weight 0.0-2.0>,
  "right_defense": <weight 0.0-2.0>,
  "reasoning": "<brief explanation>"
}}

Generate priority weights:
"""

        return prompt

    def _generate_with_agent(self, prompt: str) -> str:
        """Generate reasoning using agent model."""
        try:
            # Use agent's tokenizer and model
            inputs = self.agent.tokenizer(prompt, return_tensors="pt")

            if self.agent.model.device != 'cpu':
                inputs = inputs.to(self.agent.model.device)

            # Generate
            import torch
            gen_config = self.agent.config.get('generation', {})

            with torch.no_grad():
                outputs = self.agent.model.generate(
                    **inputs,
                    max_new_tokens=min(gen_config.get('max_new_tokens', 512), 256),
                    temperature=0.5,  # Lower temperature for more focused output
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.agent.tokenizer.eos_token_id
                )

            # Decode
            output = self.agent.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt
            output = output[len(prompt):].strip()

            return output

        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            return self._simulate_reasoning([])

    def _simulate_reasoning(self, plays: List[Dict[str, Any]]) -> str:
        """Simulate agent reasoning for testing."""
        # Analyze play patterns to simulate intelligent reasoning
        if len(plays) > 0:
            zones = [p.get('target_zone', 'center_attack') for p in plays]
            zone_counts = {}
            for zone in zones:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

            # Find most common zone
            if zone_counts:
                dominant_zone = max(zone_counts, key=zone_counts.get)
                dominant_count = zone_counts[dominant_zone]
                total = len(zones)

                if dominant_count / total > 0.6:
                    # Strong bias detected
                    return f"""
Step 1: Entropy Analysis
- High entropy concentrated in {dominant_zone} zone ({dominant_count}/{total} plays)
- Systematic uncertainty pattern detected
- Model under-confident in team's preferred attack pattern

Step 2: Team Bias Detection
- Clear preference for {dominant_zone} attacks ({dominant_count / total:.1%})
- Other zones under-represented in high-entropy samples
- Team has specific tactical pattern that needs emphasis

Step 3: Priority Weights
{{
  "left_attack": {1.8 if 'left' in dominant_zone else 1.0},
  "right_attack": {1.8 if 'right' in dominant_zone else 1.0},
  "center_attack": {1.8 if 'center' in dominant_zone else 1.0},
  "left_defense": 1.0,
  "right_defense": 1.0,
  "reasoning": "Focus on {dominant_zone} - team's primary attack pattern with high uncertainty"
}}
"""

        # Default balanced reasoning
        return """
Step 1: Entropy Analysis
- Entropy distributed across multiple zones
- No single pattern dominates uncertainty
- Mix of tactical variations

Step 2: Team Bias Detection
- Slight left-side preference observed (55%)
- Balanced attack distribution overall
- Some emphasis on left flank worth exploring

Step 3: Priority Weights
{
  "left_attack": 1.5,
  "right_attack": 1.0,
  "center_attack": 1.2,
  "left_defense": 1.0,
  "right_defense": 1.0,
  "reasoning": "Moderate focus on left attack based on observed patterns"
}
"""

    def _parse_weights(self, output: str) -> Dict[str, float]:
        """Parse weights from agent output."""
        weights = {}

        try:
            # Try to extract JSON block
            import re
            json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                # Clean up for parsing
                json_str = re.sub(r'<[^>]+>', '1.0', json_str)  # Replace placeholders
                parsed = json.loads(json_str)

                # Extract numeric weights
                for key in ['left_attack', 'right_attack', 'center_attack',
                           'left_defense', 'right_defense']:
                    if key in parsed:
                        val = parsed[key]
                        if isinstance(val, (int, float)):
                            weights[key] = float(val)
                        elif isinstance(val, str):
                            try:
                                weights[key] = float(val)
                            except:
                                weights[key] = 1.0

                # Store reasoning if present
                if 'reasoning' in parsed:
                    weights['reasoning'] = parsed['reasoning']

        except Exception as e:
            logger.warning(f"Failed to parse weights from output: {e}")

        # Ensure all zones have weights
        default_weights = self._get_default_weights()
        for key in default_weights:
            if key not in weights:
                weights[key] = default_weights[key]

        # Normalize weights to reasonable range [0.5, 2.0]
        for key in weights:
            if key != 'reasoning' and isinstance(weights[key], (int, float)):
                weights[key] = max(0.5, min(2.0, weights[key]))

        return weights

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default balanced weights."""
        return {
            'left_attack': 1.0,
            'right_attack': 1.0,
            'center_attack': 1.0,
            'left_defense': 1.0,
            'right_defense': 1.0,
            'reasoning': 'Default balanced weights'
        }

    def compute_sample_weights(
        self,
        plays: List[Dict[str, Any]],
        zone_weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute per-sample weights for LoRA training.

        Args:
            plays: Training plays with zone labels
            zone_weights: Zone priority weights from reasoning

        Returns:
            Array of sample weights
        """
        weights = np.ones(len(plays))

        for i, play in enumerate(plays):
            zone = play.get('target_zone', 'center_attack')

            # Apply zone weight if available
            if zone in zone_weights:
                weights[i] = zone_weights[zone]

        # Normalize to mean=1.0
        weights = weights / np.mean(weights)

        return weights


def reason_calibration(
    plays_json: List[Dict[str, Any]],
    entropy_threshold: float = 0.8
) -> Dict[str, float]:
    """
    Convenience function for calibration reasoning.

    Args:
        plays_json: List of plays with entropy scores
        entropy_threshold: Minimum entropy to trigger reasoning

    Returns:
        Dictionary of zone weights
    """
    reasoner = CalibrationReasoner()
    return reasoner.reason_calibration(plays_json, entropy_threshold)
