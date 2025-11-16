"""
Exploitation Heatmap for FieldSense AI v3.0
Computes opportunity from uncertainty + pressure for decision-making
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path


def compute_entropy(xt_grid: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute entropy of xT predictions (uncertainty measure).

    Args:
        xt_grid: xT probability grid of shape (H, W)
        epsilon: Small constant to avoid log(0)

    Returns:
        Entropy grid of shape (H, W)
    """
    # Clip probabilities to avoid log(0)
    p = np.clip(xt_grid, epsilon, 1 - epsilon)

    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return entropy


def compute_pressure_decay(
    player_positions: List[Dict[str, float]],
    grid_shape: Tuple[int, int] = (105, 68),
    decay_rate: float = 0.1
) -> np.ndarray:
    """
    Compute defensive pressure decay across field.

    Args:
        player_positions: List of player dicts with x, y, role, possession
        grid_shape: Field dimensions (H, W)
        decay_rate: Pressure decay rate (higher = faster decay)

    Returns:
        Pressure grid of shape (H, W)
    """
    pressure = np.zeros(grid_shape, dtype=np.float32)

    # Find defensive players (non-possession)
    for player in player_positions:
        if not player.get('possession', False):
            x = int(np.clip(player['x'], 0, grid_shape[0] - 1))
            y = int(np.clip(player['y'], 0, grid_shape[1] - 1))

            # Create pressure gradient around defender
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    dist = np.sqrt((i - x)**2 + (j - y)**2)
                    pressure[i, j] += np.exp(-decay_rate * dist)

    # Normalize to [0, 1]
    if pressure.max() > 0:
        pressure = pressure / pressure.max()

    # Invert: low pressure = high opportunity
    pressure = 1 - pressure

    return pressure


def compute_off_ball_bonus(
    player_positions: List[Dict[str, float]],
    grid_shape: Tuple[int, int] = (105, 68),
    bonus_radius: float = 15.0
) -> np.ndarray:
    """
    Compute off-ball movement bonus zones.

    Args:
        player_positions: List of player dicts
        grid_shape: Field dimensions
        bonus_radius: Radius for off-ball bonus

    Returns:
        Bonus grid of shape (H, W)
    """
    bonus = np.ones(grid_shape, dtype=np.float32)

    # Find attacking players without possession
    for player in player_positions:
        role = player.get('role', '')
        has_possession = player.get('possession', False)

        # Offensive roles moving off-ball
        is_offensive = any(r in role for r in ['Forward', 'WR', 'TE', 'Mid'])

        if is_offensive and not has_possession:
            x = int(np.clip(player['x'], 0, grid_shape[0] - 1))
            y = int(np.clip(player['y'], 0, grid_shape[1] - 1))

            # Add bonus around off-ball runners
            for i in range(max(0, x - int(bonus_radius)),
                          min(grid_shape[0], x + int(bonus_radius))):
                for j in range(max(0, y - int(bonus_radius)),
                              min(grid_shape[1], y + int(bonus_radius))):
                    dist = np.sqrt((i - x)**2 + (j - y)**2)
                    if dist < bonus_radius:
                        bonus[i, j] += 0.3 * (1 - dist / bonus_radius)

    return bonus


def compute_xt_gradient(xt_grid: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude of xT (indicates threat change rate).

    Args:
        xt_grid: xT probability grid

    Returns:
        Gradient magnitude grid
    """
    # Compute gradients using numpy
    grad_x = np.gradient(xt_grid, axis=0)
    grad_y = np.gradient(xt_grid, axis=1)

    # Magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize
    if grad_magnitude.max() > 0:
        grad_magnitude = grad_magnitude / grad_magnitude.max()

    return grad_magnitude


def compute_opportunity_heatmap(
    xt_grid: np.ndarray,
    player_positions: List[Dict[str, float]],
    weights: Dict[str, float] = None
) -> np.ndarray:
    """
    Compute exploitation opportunity heatmap.

    Formula: opportunity = (1-entropy) * pressure_decay * off_ball_bonus * xT_grad

    Args:
        xt_grid: xT predictions of shape (H, W)
        player_positions: List of player dictionaries
        weights: Optional weights for each component

    Returns:
        Opportunity heatmap of shape (H, W)
    """
    if weights is None:
        weights = {
            'certainty': 0.3,     # (1 - entropy)
            'pressure': 0.25,      # Low defensive pressure
            'off_ball': 0.2,       # Off-ball movement
            'gradient': 0.25       # xT change rate
        }

    # Compute components
    entropy = compute_entropy(xt_grid)
    certainty = 1 - entropy  # High certainty = low entropy

    pressure = compute_pressure_decay(player_positions)
    off_ball = compute_off_ball_bonus(player_positions)
    gradient = compute_xt_gradient(xt_grid)

    # Combine with weights
    opportunity = (
        weights['certainty'] * certainty +
        weights['pressure'] * pressure +
        weights['off_ball'] * off_ball +
        weights['gradient'] * gradient
    )

    # Normalize to [0, 1]
    if opportunity.max() > opportunity.min():
        opportunity = (opportunity - opportunity.min()) / (opportunity.max() - opportunity.min())

    # Apply smoothing
    opportunity = gaussian_filter(opportunity, sigma=2.0)

    return opportunity


def find_top_decision(
    opportunity: np.ndarray,
    xt_grid: np.ndarray,
    player_positions: List[Dict[str, float]],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Find top decision from opportunity heatmap.

    Args:
        opportunity: Opportunity heatmap
        xt_grid: xT predictions
        player_positions: Player positions
        threshold: Minimum opportunity threshold

    Returns:
        Decision dictionary with text, location, confidence
    """
    # Find peak opportunity zone
    max_idx = np.unravel_index(np.argmax(opportunity), opportunity.shape)
    max_opp = opportunity[max_idx]
    max_xt = xt_grid[max_idx]

    # Determine zone (left/center/right)
    y_pos = max_idx[1]
    if y_pos < 23:
        zone = "left"
    elif y_pos < 45:
        zone = "center"
    else:
        zone = "right"

    # Determine action based on xT and position
    if max_xt > 0.6:
        action = "Attack"
    elif max_xt > 0.4:
        action = "Advance"
    else:
        action = "Hold"

    # Compute expected improvement
    current_xt = np.mean(xt_grid)
    improvement = int((max_xt - current_xt) * 100)

    # Build decision text
    sign = "+" if improvement >= 0 else ""
    decision_text = f"{action} {zone}: {sign}{improvement}% xT"

    # Determine confidence
    entropy = compute_entropy(xt_grid)
    avg_entropy = np.mean(entropy)

    if avg_entropy < 0.3:
        confidence = "High"
    elif avg_entropy < 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        'top_decision': decision_text,
        'action': action,
        'zone': zone,
        'location': {'x': int(max_idx[0]), 'y': int(max_idx[1])},
        'opportunity': float(max_opp),
        'xt_improvement': improvement,
        'confidence': confidence,
        'avg_entropy': float(avg_entropy)
    }


def generate_insight(
    xt_grid: np.ndarray,
    player_positions: List[Dict[str, float]],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate complete exploitation insight with heatmap.

    Args:
        xt_grid: xT predictions of shape (H, W)
        player_positions: List of player dictionaries
        output_path: Optional path to save heatmap PNG

    Returns:
        Insight dictionary with decision, zones, confidence
    """
    # Compute opportunity heatmap
    opportunity = compute_opportunity_heatmap(xt_grid, player_positions)

    # Find top decision
    decision = find_top_decision(opportunity, xt_grid, player_positions)

    # Create output
    insight = {
        'top_decision': decision['top_decision'],
        'zones': opportunity,
        'confidence': decision['confidence'],
        'details': decision
    }

    # Save visualization if requested
    if output_path:
        save_opportunity_heatmap(
            opportunity,
            output_path,
            title=decision['top_decision'],
            confidence=decision['confidence']
        )

    return insight


def save_opportunity_heatmap(
    opportunity: np.ndarray,
    output_path: str,
    title: str = "Exploitation Opportunity",
    confidence: str = "Medium"
):
    """
    Save opportunity heatmap as green-gold PNG.

    Args:
        opportunity: Opportunity grid (H, W)
        output_path: Path to save PNG
        title: Plot title
        confidence: Confidence level
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot heatmap with green-gold colormap
    im = ax.imshow(
        opportunity.T,
        cmap='YlGn',  # Yellow-Green (green-gold)
        aspect='auto',
        origin='lower',
        extent=[0, 105, 0, 68],
        vmin=0,
        vmax=1
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Opportunity Score', rotation=270, labelpad=20)

    # Title with confidence
    ax.set_title(f"{title}\nConfidence: {confidence}",
                fontsize=18, fontweight='bold', pad=20)

    # Labels
    ax.set_xlabel('Field Length (m)', fontsize=12)
    ax.set_ylabel('Field Width (m)', fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add zone markers
    ax.axvline(x=52.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=34, color='white', linestyle='--', alpha=0.5, linewidth=1)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Opportunity heatmap saved to {output_path}")
