"""
Physics-Constrained Counterfactuals for FieldSense AI v3.1
Simulates alternative play scenarios with realistic constraints
Enhanced with MiroThinker agent verification for improved validity
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy

# Agent integration for counterfactual verification
try:
    from ..agent import CounterfactualReasoner
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    logging.warning("Agent not available - counterfactuals will run without agent verification")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Field constraints
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
MAX_PLAYER_SPEED = 11.0  # m/s (elite sprint speed)
BALL_DECAY_RATE = 0.95  # Velocity decay per time step
COLLISION_RADIUS = 2.0   # meters
OFFSIDE_THRESHOLD = 5.0  # meters (simplified)


@dataclass
class Perturbation:
    """Represents a player perturbation."""
    player_id: str
    delay: float = 0.0        # seconds
    speed_factor: float = 1.0  # multiplier
    angle_delta: float = 0.0   # degrees


@dataclass
class CounterfactualResult:
    """Counterfactual simulation result with agent verification."""
    player: str
    change: str
    lift: float  # delta xT (refined by agent if available)
    valid: bool
    violation_reason: Optional[str] = None
    new_positions: Optional[List[Dict[str, float]]] = None
    # Agent verification fields (v3.1)
    cvs: Optional[float] = None  # Counterfactual Validity Score
    agent_trace: Optional[str] = None  # Agent reasoning trace
    chain_steps: Optional[int] = None  # Number of reasoning steps
    iterations: Optional[int] = None  # Re-chain iterations


class PhysicsSimulator:
    """Simulates ball and player physics with realistic constraints."""

    def __init__(
        self,
        dt: float = 0.1,  # time step in seconds
        gravity: float = 9.81,
        ball_mass: float = 0.45,  # kg
        air_resistance: float = 0.02
    ):
        """
        Initialize physics simulator.

        Args:
            dt: Time step for simulation
            gravity: Gravitational acceleration
            ball_mass: Ball mass in kg
            air_resistance: Air resistance coefficient
        """
        self.dt = dt
        self.gravity = gravity
        self.ball_mass = ball_mass
        self.air_resistance = air_resistance

    def simulate_player_path(
        self,
        start_pos: np.ndarray,
        velocity: float,
        direction: float,
        duration: float,
        perturbation: Optional[Perturbation] = None
    ) -> List[np.ndarray]:
        """
        Simulate player movement path.

        Args:
            start_pos: Starting position [x, y]
            velocity: Initial velocity (m/s)
            direction: Movement direction (degrees)
            duration: Simulation duration (seconds)
            perturbation: Optional perturbation to apply

        Returns:
            List of positions over time
        """
        positions = [start_pos.copy()]
        current_pos = start_pos.copy()

        # Apply perturbations
        if perturbation:
            velocity *= perturbation.speed_factor
            direction += perturbation.angle_delta

        # Clamp velocity to realistic bounds
        velocity = np.clip(velocity, 0, MAX_PLAYER_SPEED)

        # Convert direction to radians
        dir_rad = np.radians(direction)
        vel_vec = np.array([
            velocity * np.cos(dir_rad),
            velocity * np.sin(dir_rad)
        ])

        # Simulate movement
        steps = int(duration / self.dt)
        delay_steps = 0

        if perturbation and perturbation.delay > 0:
            delay_steps = int(perturbation.delay / self.dt)

        for step in range(steps):
            if step < delay_steps:
                # Player is delayed, stay in place
                positions.append(current_pos.copy())
            else:
                # Move with velocity
                current_pos = current_pos + vel_vec * self.dt

                # Keep in bounds
                current_pos[0] = np.clip(current_pos[0], 0, FIELD_LENGTH)
                current_pos[1] = np.clip(current_pos[1], 0, FIELD_WIDTH)

                positions.append(current_pos.copy())

        return positions

    def simulate_ball_trajectory(
        self,
        start_pos: np.ndarray,
        initial_velocity: np.ndarray,
        duration: float
    ) -> List[np.ndarray]:
        """
        Simulate ball trajectory with momentum conservation and decay.

        Args:
            start_pos: Starting position [x, y]
            initial_velocity: Initial velocity vector [vx, vy]
            duration: Simulation duration

        Returns:
            List of ball positions over time
        """
        positions = [start_pos.copy()]
        current_pos = start_pos.copy()
        current_vel = initial_velocity.copy()

        steps = int(duration / self.dt)

        for step in range(steps):
            # Apply air resistance (velocity decay)
            current_vel *= BALL_DECAY_RATE

            # Apply pass falloff (distance-dependent decay)
            speed = np.linalg.norm(current_vel)
            if speed < 0.5:
                current_vel *= 0.9  # Rapid decay at low speeds

            # Update position
            current_pos = current_pos + current_vel * self.dt

            # Bounce off field boundaries
            if current_pos[0] < 0 or current_pos[0] > FIELD_LENGTH:
                current_vel[0] *= -0.5  # Bounce with energy loss
                current_pos[0] = np.clip(current_pos[0], 0, FIELD_LENGTH)

            if current_pos[1] < 0 or current_pos[1] > FIELD_WIDTH:
                current_vel[1] *= -0.5
                current_pos[1] = np.clip(current_pos[1], 0, FIELD_WIDTH)

            positions.append(current_pos.copy())

        return positions


class CounterfactualGenerator:
    """Generates and validates counterfactual scenarios with agent verification."""

    def __init__(
        self,
        simulator: Optional[PhysicsSimulator] = None,
        use_agent: bool = True,
        cvs_threshold: float = 0.95
    ):
        """
        Initialize counterfactual generator.

        Args:
            simulator: Physics simulator (default: PhysicsSimulator())
            use_agent: Enable agent verification (default: True)
            cvs_threshold: Minimum CVS for agent verification (default: 0.95)
        """
        self.simulator = simulator or PhysicsSimulator()
        self.use_agent = use_agent
        self.cvs_threshold = cvs_threshold

        # Initialize agent if requested
        self.agent_reasoner = None
        if self.use_agent and AGENT_AVAILABLE:
            try:
                self.agent_reasoner = CounterfactualReasoner(
                    cvs_threshold=cvs_threshold,
                    max_retries=3
                )
                logger.info("Agent verification enabled for counterfactuals")
            except Exception as e:
                logger.warning(f"Failed to initialize agent: {e}")
                self.agent_reasoner = None

    def generate_perturbations(
        self,
        play_data: Dict[str, Any],
        n_perturbations: int = 3
    ) -> List[Perturbation]:
        """
        Generate perturbations for top players.

        Args:
            play_data: Original play data
            n_perturbations: Number of perturbations to generate

        Returns:
            List of perturbations
        """
        players = play_data.get('players', [])

        # Focus on offensive players with high velocity
        candidates = []
        for player in players:
            if any(role in player.get('role', '') for role in ['WR', 'Forward', 'TE']):
                candidates.append(player)

        # Sort by velocity (more dynamic players)
        candidates.sort(key=lambda p: p.get('vel', 0), reverse=True)

        perturbations = []

        # Generate diverse perturbations for each candidate
        variation_templates = [
            {'delay': 0.5, 'speed_factor': 1.0, 'angle_delta': 0.0},     # Delay
            {'delay': 0.0, 'speed_factor': 0.9, 'angle_delta': 0.0},     # Slower
            {'delay': 0.0, 'speed_factor': 1.0, 'angle_delta': 15.0},    # Angle change
            {'delay': 0.3, 'speed_factor': 0.95, 'angle_delta': -10.0},  # Combined
        ]

        for i, player in enumerate(candidates[:n_perturbations]):
            if i < len(variation_templates):
                template = variation_templates[i]
                pert = Perturbation(
                    player_id=player['id'],
                    delay=template['delay'],
                    speed_factor=template['speed_factor'],
                    angle_delta=template['angle_delta']
                )
                perturbations.append(pert)

        return perturbations

    def simulate_counterfactual(
        self,
        play_data: Dict[str, Any],
        perturbation: Perturbation,
        duration: float = 2.0
    ) -> Dict[str, Any]:
        """
        Simulate a counterfactual scenario.

        Args:
            play_data: Original play data
            perturbation: Perturbation to apply
            duration: Simulation duration

        Returns:
            Simulated play data
        """
        simulated_play = deepcopy(play_data)

        # Find player to perturb
        target_player = None
        for player in simulated_play['players']:
            if player['id'] == perturbation.player_id:
                target_player = player
                break

        if not target_player:
            return simulated_play

        # Simulate player path
        start_pos = np.array([target_player['x'], target_player['y']])
        new_positions = self.simulator.simulate_player_path(
            start_pos=start_pos,
            velocity=target_player.get('vel', 5.0),
            direction=target_player.get('dir', 0.0),
            duration=duration,
            perturbation=perturbation
        )

        # Update player position to final position
        if new_positions:
            final_pos = new_positions[-1]
            target_player['x'] = float(final_pos[0])
            target_player['y'] = float(final_pos[1])

        # Store trajectory for visualization
        simulated_play['trajectory'] = {
            'player_id': perturbation.player_id,
            'positions': new_positions
        }

        # Simulate ball if player has possession
        if target_player.get('possession', False):
            ball_start = np.array([simulated_play['ball']['x'], simulated_play['ball']['y']])
            ball_vel = np.array([
                target_player.get('vel', 5.0) * np.cos(np.radians(target_player.get('dir', 0))),
                target_player.get('vel', 5.0) * np.sin(np.radians(target_player.get('dir', 0)))
            ])

            ball_positions = self.simulator.simulate_ball_trajectory(
                start_pos=ball_start,
                initial_velocity=ball_vel,
                duration=duration
            )

            if ball_positions:
                final_ball = ball_positions[-1]
                simulated_play['ball']['x'] = float(final_ball[0])
                simulated_play['ball']['y'] = float(final_ball[1])

        return simulated_play

    def validate_scenario(
        self,
        simulated_play: Dict[str, Any],
        original_play: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate counterfactual scenario against physics constraints.

        Args:
            simulated_play: Simulated play data
            original_play: Original play data

        Returns:
            (is_valid, violation_reason)
        """
        # Check ball in bounds
        ball_x = simulated_play['ball']['x']
        ball_y = simulated_play['ball']['y']

        if ball_x < 0 or ball_x > FIELD_LENGTH or ball_y < 0 or ball_y > FIELD_WIDTH:
            return False, "Ball out of bounds"

        # Check all players in bounds
        for player in simulated_play['players']:
            if player['x'] < 0 or player['x'] > FIELD_LENGTH:
                return False, f"Player {player['id']} out of bounds (x)"
            if player['y'] < 0 or player['y'] > FIELD_WIDTH:
                return False, f"Player {player['id']} out of bounds (y)"

        # Check for collisions
        players = simulated_play['players']
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                dist = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                if dist < COLLISION_RADIUS:
                    return False, f"Collision between {p1['id']} and {p2['id']}"

        # Simplified offside check
        # (In real implementation, would check relative to defenders and ball)
        offensive_players = [p for p in players if 'Forward' in p.get('role', '') or 'WR' in p.get('role', '')]
        if offensive_players:
            max_x = max(p['x'] for p in offensive_players)
            if max_x > FIELD_LENGTH - OFFSIDE_THRESHOLD:
                # Check if too far ahead of ball
                if max_x > ball_x + OFFSIDE_THRESHOLD:
                    return False, "Potential offside violation"

        return True, None

    def compute_delta_xt(
        self,
        original_xt: float,
        simulated_xt: float
    ) -> float:
        """
        Compute change in xT value.

        Args:
            original_xt: Original xT value
            simulated_xt: Simulated xT value

        Returns:
            Delta xT (lift)
        """
        return simulated_xt - original_xt

    def compute_cvs(self, results: List[CounterfactualResult]) -> float:
        """
        Compute Counterfactual Validity Score (CVS).

        CVS = (# valid scenarios) / (total scenarios)

        Args:
            results: List of counterfactual results

        Returns:
            CVS score (0-1)
        """
        if not results:
            return 0.0

        valid_count = sum(1 for r in results if r.valid)
        return valid_count / len(results)

    def generate_counterfactuals(
        self,
        play_data: Dict[str, Any],
        original_xt: float = 0.5,
        n_scenarios: int = 3
    ) -> Tuple[List[CounterfactualResult], float]:
        """
        Generate multiple counterfactual scenarios with agent verification.

        Args:
            play_data: Original play data
            original_xt: Original xT value
            n_scenarios: Number of scenarios to generate

        Returns:
            (List of counterfactual results, CVS score)
        """
        perturbations = self.generate_perturbations(play_data, n_scenarios)
        results = []

        for pert in perturbations:
            # Simulate scenario
            simulated_play = self.simulate_counterfactual(
                play_data,
                pert,
                duration=2.0
            )

            # Validate
            is_valid, violation = self.validate_scenario(simulated_play, play_data)

            # Compute simulated xT (simplified - would use actual model)
            # For now, estimate based on ball position
            ball_x = simulated_play['ball']['x']
            simulated_xt = original_xt + (ball_x - play_data['ball']['x']) / FIELD_LENGTH * 0.3

            # Compute delta
            delta_xt = self.compute_delta_xt(original_xt, simulated_xt)

            # Create change description
            if pert.delay > 0:
                change = f"Delay {pert.delay:.1f}s"
            elif abs(pert.speed_factor - 1.0) > 0.01:
                pct = int((pert.speed_factor - 1.0) * 100)
                change = f"Speed {pct:+d}%"
            elif abs(pert.angle_delta) > 0:
                change = f"Angle {pert.angle_delta:+.0f}°"
            else:
                change = "No change"

            # Find player
            player_name = pert.player_id
            for p in play_data['players']:
                if p['id'] == pert.player_id:
                    player_name = p.get('role', pert.player_id)
                    break

            # Agent verification (if enabled)
            agent_cvs = None
            agent_trace = None
            chain_steps = None
            iterations = None
            refined_lift = delta_xt

            if self.agent_reasoner is not None:
                # Prepare perturbation and simulation result for agent
                pert_dict = {
                    'player_id': pert.player_id,
                    'delay': pert.delay,
                    'speed_factor': pert.speed_factor,
                    'angle_delta': pert.angle_delta
                }

                sim_result = {
                    'valid': is_valid,
                    'violation_reason': violation,
                    'lift': delta_xt,
                    'ball_x': ball_x,
                    'simulated_xt': simulated_xt
                }

                try:
                    # Call agent verification with re-chain logic
                    agent_result = self.agent_reasoner.verify_and_refine(pert_dict, sim_result)

                    # Extract agent results
                    refined_lift = agent_result.get('lift', delta_xt)
                    agent_cvs = agent_result.get('cvs', None)
                    agent_trace = agent_result.get('trace', None)
                    chain_steps = agent_result.get('chain_steps', None)
                    iterations = agent_result.get('iterations', 1)

                    # Update validity based on agent CVS
                    if agent_cvs is not None and agent_cvs < self.cvs_threshold:
                        is_valid = False
                        if not violation:
                            violation = f"Low agent confidence (CVS: {agent_cvs:.3f})"

                except Exception as e:
                    logger.warning(f"Agent verification failed: {e}")

            result = CounterfactualResult(
                player=player_name,
                change=change,
                lift=refined_lift,
                valid=is_valid,
                violation_reason=violation,
                new_positions=simulated_play.get('trajectory', {}).get('positions', []),
                cvs=agent_cvs,
                agent_trace=agent_trace,
                chain_steps=chain_steps,
                iterations=iterations
            )

            results.append(result)

        # Compute CVS (aggregate from agent if available, else physics-based)
        if self.agent_reasoner is not None and any(r.cvs is not None for r in results):
            # Use agent CVS scores
            cvs_scores = [r.cvs for r in results if r.cvs is not None]
            cvs = float(np.mean(cvs_scores)) if cvs_scores else self.compute_cvs(results)
        else:
            # Use physics-based CVS
            cvs = self.compute_cvs(results)

        return results, cvs


def create_counterfactual_payload(
    play_data: Dict[str, Any],
    original_xt: float,
    n_scenarios: int = 3,
    use_agent: bool = True,
    lightweight: bool = True
) -> Dict[str, Any]:
    """
    Create counterfactual payload for UI with agent verification.

    Args:
        play_data: Original play data
        original_xt: Original xT value
        n_scenarios: Number of scenarios
        use_agent: Enable agent verification (default: True)
        lightweight: Distill to lightweight output <1MB (default: True)

    Returns:
        Payload with counterfactuals and CVS
    """
    generator = CounterfactualGenerator(use_agent=use_agent)
    results, cvs = generator.generate_counterfactuals(
        play_data,
        original_xt,
        n_scenarios
    )

    # Format for UI
    counterfactuals = []
    for result in results:
        cf = {
            'player': result.player,
            'change': result.change,
            'lift': round(result.lift, 3),
            'valid': result.valid,
            'violation': result.violation_reason
        }

        # Add agent verification fields (v3.1)
        if result.cvs is not None:
            cf['cvs'] = round(result.cvs, 3)

        if result.chain_steps is not None:
            cf['chain_steps'] = result.chain_steps

        if result.iterations is not None:
            cf['iterations'] = result.iterations

        # Add agent trace (truncate if lightweight mode)
        if result.agent_trace:
            if lightweight and len(result.agent_trace) > 200:
                cf['agent_trace'] = result.agent_trace[:197] + "..."
            else:
                cf['agent_trace'] = result.agent_trace

        # Add trajectory if available (sample if lightweight mode)
        if result.new_positions:
            if lightweight and len(result.new_positions) > 20:
                # Sample every nth position to reduce size
                step = len(result.new_positions) // 20
                sampled = result.new_positions[::step][:20]
                cf['trajectory'] = [
                    {'x': round(float(pos[0]), 2), 'y': round(float(pos[1]), 2)}
                    for pos in sampled
                ]
            else:
                cf['trajectory'] = [
                    {'x': round(float(pos[0]), 2), 'y': round(float(pos[1]), 2)}
                    for pos in result.new_positions
                ]

        counterfactuals.append(cf)

    payload = {
        'counterfactuals': counterfactuals,
        'cvs': round(cvs, 3),
        'n_valid': sum(1 for cf in counterfactuals if cf['valid']),
        'n_total': len(counterfactuals),
        'agent_enabled': use_agent and AGENT_AVAILABLE
    }

    # Estimate payload size and warn if too large
    import sys
    payload_size = sys.getsizeof(str(payload))
    if payload_size > 1_000_000:  # 1MB
        logger.warning(f"Payload size ({payload_size:,} bytes) exceeds 1MB limit")

    return payload
