"""
Agent-Based Opportunity Chaining for FieldSense AI v3.1

Uses MiroThinker to chain opportunities and risks from xT heatmap:
- Step 1: Identify top attack zones (1st-order opportunities)
- Step 2: Analyze secondary risks (CB counter, defensive reactions)
- Outputs: Upgraded zones with risk vectors, chain depth (2-4 steps)

Turns static heatmap into dynamic decision graph.
"""

import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Try to import agent
try:
    from .load import MiroThinkerAgent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    logging.warning("MiroThinker not available - using simulation mode")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeatmapReasoner:
    """
    Agent-based reasoner for heatmap opportunity chaining.

    Analyzes xT heatmap to identify multi-step opportunities and risks.
    """

    def __init__(self, max_depth: int = 4, max_tokens: int = 256):
        """
        Initialize heatmap reasoner.

        Args:
            max_depth: Maximum chain depth (default: 4)
            max_tokens: Maximum tokens for generation (default: 256)
        """
        self.max_depth = max_depth
        self.max_tokens = max_tokens

        # Initialize agent if available
        self.agent = None
        if AGENT_AVAILABLE:
            try:
                self.agent = MiroThinkerAgent(offline_mode=True)
                logger.info("MiroThinker agent initialized for heatmap reasoning")
            except Exception as e:
                logger.warning(f"Failed to initialize agent: {e}")
                self.agent = None

    def _build_reasoning_prompt(
        self,
        grid_data: Dict[str, Any],
        top_zones: List[Dict[str, Any]]
    ) -> str:
        """
        Build reasoning prompt for agent.

        Args:
            grid_data: Grid information (shape, stats)
            top_zones: Top opportunity zones

        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this football field opportunity heatmap and chain multi-step decisions:

GRID DATA:
{json.dumps(grid_data, indent=2)}

TOP ZONES:
{json.dumps(top_zones, indent=2)}

CHAIN OPPORTUNITIES (2-4 steps):

Step 1 - Primary Opportunities:
Identify top attack zones with highest xT improvement.
- Which zones offer immediate attacking advantage?
- What is the expected xT lift percentage?

Step 2 - Secondary Risks:
Analyze defensive counters and risk vectors.
- CB (cornerback) positioning and counter threats
- Defensive reactions to primary moves
- Risk mitigation strategies

Step 3 - Dynamic Adjustments (optional):
Consider cascading effects.
- Off-ball movement opportunities
- Pressure redistribution
- Alternative paths if primary blocked

Step 4 - Final Recommendation (optional):
Synthesize multi-step plan.

OUTPUT (JSON format, max {self.max_tokens} tokens):
{{
  "upgraded_zones": {{
    "zone_name": {{
      "base_opportunity": <float>,
      "risk_vector": <float, -1 to 1>,
      "confidence": <float, 0-1>,
      "reasoning": "<brief explanation>"
    }}
  }},
  "depth": <int, 2-{self.max_depth}>,
  "primary_action": "<action description>",
  "risks": ["<risk 1>", "<risk 2>"],
  "chain_summary": "<concise multi-step plan>"
}}

Begin reasoning:"""

        return prompt

    def _simulate_reasoning(
        self,
        grid_data: Dict[str, Any],
        top_zones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulate agent reasoning (for testing without actual model).

        Args:
            grid_data: Grid information
            top_zones: Top opportunity zones

        Returns:
            Simulated reasoning output
        """
        # Determine chain depth (2-4 steps)
        depth = np.random.randint(2, self.max_depth + 1)

        # Build upgraded zones from top zones
        upgraded_zones = {}

        for i, zone in enumerate(top_zones[:3]):  # Top 3 zones
            zone_name = zone.get('zone', f'zone_{i}')
            base_opp = zone.get('opportunity', 0.5)

            # Simulate risk analysis
            # Higher opportunity zones may have higher defensive risk
            risk_vector = -0.12 if base_opp > 0.7 else -0.05

            # Add some variance
            risk_vector += np.random.uniform(-0.05, 0.05)
            risk_vector = np.clip(risk_vector, -0.5, 0.2)

            # Confidence based on uncertainty
            confidence = 0.85 + np.random.uniform(-0.10, 0.10)
            confidence = np.clip(confidence, 0.7, 0.95)

            # Generate reasoning based on zone characteristics
            if base_opp > 0.7:
                reasoning = f"High-value attack zone ({int(base_opp*100)}%) but faces CB counter risk. Recommend off-ball support."
            elif base_opp > 0.5:
                reasoning = f"Moderate opportunity ({int(base_opp*100)}%) with controlled risk. Good secondary option."
            else:
                reasoning = f"Lower priority zone ({int(base_opp*100)}%). Consider if primary blocked."

            upgraded_zones[zone_name] = {
                'base_opportunity': float(base_opp),
                'risk_vector': float(risk_vector),
                'confidence': float(confidence),
                'reasoning': reasoning
            }

        # Determine primary action
        if top_zones:
            top_zone = top_zones[0]
            improvement = top_zone.get('improvement', 0)
            zone_name = top_zone.get('zone', 'center')
            primary_action = f"Attack {zone_name} +{improvement}%"
        else:
            primary_action = "Hold position, reassess"

        # Generate risk list
        risks = []
        if any(z.get('opportunity', 0) > 0.7 for z in top_zones):
            risks.append("CB counter on high-value zone (risk: -12%)")
        if len(top_zones) > 1:
            risks.append("Defensive pressure redistribution")
        if depth >= 3:
            risks.append("Off-ball timing coordination required")

        # Build chain summary
        if depth == 2:
            chain_summary = f"{primary_action}, monitor defensive reaction"
        elif depth == 3:
            chain_summary = f"{primary_action}, prepare secondary option if countered, off-ball support ready"
        else:
            chain_summary = f"{primary_action}, multi-path execution with {len(upgraded_zones)} viable zones, dynamic adjustment ready"

        return {
            'upgraded_zones': upgraded_zones,
            'depth': int(depth),
            'primary_action': primary_action,
            'risks': risks,
            'chain_summary': chain_summary
        }

    def _parse_agent_output(self, output: str) -> Dict[str, Any]:
        """
        Parse agent output to extract structured result.

        Args:
            output: Raw agent output text

        Returns:
            Parsed result dictionary
        """
        # Try to find JSON block in output
        try:
            # Look for JSON block
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = output[start_idx:end_idx]
                result = json.loads(json_str)

                # Validate required fields
                if 'upgraded_zones' in result and 'depth' in result:
                    # Ensure depth is in valid range
                    result['depth'] = int(np.clip(result['depth'], 2, self.max_depth))
                    return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse agent output as JSON: {e}")

        # Fallback: return minimal structure
        return {
            'upgraded_zones': {},
            'depth': 2,
            'primary_action': "Parsing error",
            'risks': ['Agent output parsing failed'],
            'chain_summary': "Unable to parse agent reasoning"
        }

    def reason_opportunity(
        self,
        grid_data: Dict[str, Any],
        top_zones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run agent reasoning chain on heatmap opportunities.

        Args:
            grid_data: Grid information with shape, stats, etc.
            top_zones: Top opportunity zones from heatmap analysis

        Returns:
            Upgraded zones with risk vectors and chain depth:
            {
                'upgraded_zones': dict of zone upgrades,
                'depth': int (2-4),
                'primary_action': str,
                'risks': list of risk descriptions,
                'chain_summary': str
            }
        """
        if self.agent is None:
            # Use simulation mode
            return self._simulate_reasoning(grid_data, top_zones)

        # Build prompt
        prompt = self._build_reasoning_prompt(grid_data, top_zones)

        # Generate reasoning
        try:
            # Use agent's generation method
            output = self.agent._generate(
                prompt,
                max_new_tokens=self.max_tokens
            )

            # Parse output
            result = self._parse_agent_output(output)

            return result

        except Exception as e:
            logger.error(f"Agent reasoning failed: {e}")
            # Fallback to simulation
            return self._simulate_reasoning(grid_data, top_zones)

    def chain_zones(
        self,
        opportunity_grid: np.ndarray,
        xt_grid: np.ndarray,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Chain opportunities from heatmap grids.

        Args:
            opportunity_grid: Opportunity heatmap (H, W)
            xt_grid: xT predictions (H, W)
            top_n: Number of top zones to analyze (default: 3)

        Returns:
            Chained opportunity analysis
        """
        # Find top N zones
        flat_indices = np.argsort(opportunity_grid.flatten())[-top_n:][::-1]
        top_zones = []

        for idx in flat_indices:
            coord = np.unravel_index(idx, opportunity_grid.shape)
            opp_value = opportunity_grid[coord]
            xt_value = xt_grid[coord]

            # Determine zone name
            y_pos = coord[1]
            if y_pos < 23:
                zone_name = "left"
            elif y_pos < 45:
                zone_name = "center"
            else:
                zone_name = "right"

            # Compute improvement vs average
            avg_xt = np.mean(xt_grid)
            improvement = int((xt_value - avg_xt) * 100)

            top_zones.append({
                'zone': zone_name,
                'location': {'x': int(coord[0]), 'y': int(coord[1])},
                'opportunity': float(opp_value),
                'xt': float(xt_value),
                'improvement': improvement
            })

        # Prepare grid data
        grid_data = {
            'shape': list(opportunity_grid.shape),
            'mean_opportunity': float(np.mean(opportunity_grid)),
            'max_opportunity': float(np.max(opportunity_grid)),
            'mean_xt': float(np.mean(xt_grid)),
            'n_zones': top_n
        }

        # Run agent reasoning
        result = self.reason_opportunity(grid_data, top_zones)

        return result


def reason_opportunity(
    grid_data: Dict[str, Any],
    top_zones: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convenience function for single opportunity reasoning.

    Args:
        grid_data: Grid information
        top_zones: Top opportunity zones

    Returns:
        Upgraded zones with risk analysis
    """
    reasoner = HeatmapReasoner()
    return reasoner.reason_opportunity(grid_data, top_zones)


def chain_heatmap_opportunities(
    opportunity_grid: np.ndarray,
    xt_grid: np.ndarray,
    enable_agent: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Chain opportunities from heatmap with agent analysis.

    Args:
        opportunity_grid: Opportunity heatmap (H, W)
        xt_grid: xT predictions (H, W)
        enable_agent: Enable agent reasoning (default: True)

    Returns:
        Chained analysis or None if disabled
    """
    if not enable_agent:
        return None

    reasoner = HeatmapReasoner()
    return reasoner.chain_zones(opportunity_grid, xt_grid, top_n=3)
