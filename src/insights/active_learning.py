"""
Active Learning for FieldSense AI v3.0
Flags low-confidence plays and collects human labels for continuous improvement
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class ActiveLearningManager:
    """
    Manages active learning loop: flag uncertain plays, collect labels, improve model.
    """

    def __init__(
        self,
        labels_path: str = "outputs/active_labels.json",
        entropy_threshold: float = 0.8
    ):
        """
        Initialize active learning manager.

        Args:
            labels_path: Path to save/load labels
            entropy_threshold: Entropy threshold for flagging (>0.8 = uncertain)
        """
        self.labels_path = Path(labels_path)
        self.entropy_threshold = entropy_threshold
        self.labels = self._load_labels()

    def _load_labels(self) -> List[Dict[str, Any]]:
        """Load existing labels from file."""
        if self.labels_path.exists():
            try:
                with open(self.labels_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_labels(self):
        """Save labels to file."""
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels_path, 'w') as f:
            json.dump(self.labels, f, indent=2)

    def should_flag_for_review(
        self,
        xt_grid: np.ndarray,
        insight: Dict[str, Any]
    ) -> bool:
        """
        Determine if play should be flagged for human review.

        Args:
            xt_grid: xT predictions
            insight: Generated insight dictionary

        Returns:
            True if play should be flagged for review
        """
        # Check entropy threshold
        avg_entropy = insight.get('details', {}).get('avg_entropy', 0)

        if avg_entropy > self.entropy_threshold:
            return True

        # Check confidence level
        if insight.get('confidence', '') == 'Low':
            return True

        # Check opportunity variance
        zones = insight.get('zones')
        if zones is not None:
            zone_variance = np.var(zones)
            if zone_variance < 0.01:  # Too uniform = uncertain
                return True

        return False

    def create_review_prompt(
        self,
        play_data: Dict[str, Any],
        insight: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create review prompt for human labeler.

        Args:
            play_data: Original play data (normalized frame)
            insight: Generated insight

        Returns:
            Review prompt dictionary
        """
        prompt = {
            'play_id': play_data.get('play_id', f"play_{len(self.labels)}"),
            'timestamp': datetime.now().isoformat(),
            'decision': insight['top_decision'],
            'confidence': insight['confidence'],
            'avg_entropy': insight['details']['avg_entropy'],
            'play_snapshot': {
                'time': play_data.get('time', 0),
                'n_players': len(play_data.get('players', [])),
                'ball_position': play_data.get('ball', {})
            },
            'questions': [
                {
                    'id': 'outcome',
                    'text': 'What was the actual outcome?',
                    'options': ['Success', 'Fail', 'Intercepted'],
                    'required': True
                },
                {
                    'id': 'tags',
                    'text': 'Add descriptive tags (comma-separated)',
                    'type': 'text',
                    'examples': ['LB late', 'CB overcommit', 'Safety blitz']
                }
            ],
            'flagged_reason': self._get_flag_reason(insight)
        }

        return prompt

    def _get_flag_reason(self, insight: Dict[str, Any]) -> str:
        """Get human-readable reason for flagging."""
        avg_entropy = insight['details']['avg_entropy']

        if avg_entropy > 0.8:
            return f"High uncertainty (entropy: {avg_entropy:.2f})"
        elif insight['confidence'] == 'Low':
            return "Low confidence prediction"
        else:
            return "Borderline decision"

    def submit_label(
        self,
        play_id: str,
        outcome: str,
        tags: List[str] = None,
        notes: str = "",
        play_data: Dict[str, Any] = None,
        insight: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Submit human label for a flagged play.

        Args:
            play_id: Unique play identifier
            outcome: One of 'Success', 'Fail', 'Intercepted'
            tags: Optional list of descriptive tags
            notes: Optional free-form notes
            play_data: Original play data
            insight: Generated insight

        Returns:
            Saved label dictionary
        """
        label = {
            'play_id': play_id,
            'timestamp': datetime.now().isoformat(),
            'outcome': outcome,
            'tags': tags or [],
            'notes': notes,
            'play_data': play_data,
            'predicted_decision': insight.get('top_decision') if insight else None,
            'predicted_confidence': insight.get('confidence') if insight else None
        }

        # Add to labels list
        self.labels.append(label)

        # Save to file
        self._save_labels()

        print(f"Label saved: {play_id} -> {outcome}")

        return label

    def get_labels_for_calibration(
        self,
        min_labels: int = 10,
        outcome_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get labeled plays for model calibration.

        Args:
            min_labels: Minimum number of labels required
            outcome_filter: Optional filter by outcome

        Returns:
            List of labeled plays suitable for calibration
        """
        filtered_labels = self.labels

        # Filter by outcome if specified
        if outcome_filter:
            filtered_labels = [
                l for l in filtered_labels
                if l['outcome'] == outcome_filter
            ]

        # Check minimum threshold
        if len(filtered_labels) < min_labels:
            print(f"Warning: Only {len(filtered_labels)} labels available "
                  f"(minimum {min_labels} recommended)")

        return filtered_labels

    def export_labels_for_training(
        self,
        output_path: str,
        include_failures: bool = True
    ) -> Dict[str, Any]:
        """
        Export labels in format suitable for model training.

        Args:
            output_path: Path to save training data
            include_failures: Whether to include failed plays

        Returns:
            Training data dictionary
        """
        training_data = {
            'plays': [],
            'metadata': {
                'n_labels': len(self.labels),
                'exported_at': datetime.now().isoformat(),
                'entropy_threshold': self.entropy_threshold
            }
        }

        for label in self.labels:
            # Skip failures if not included
            if not include_failures and label['outcome'] == 'Fail':
                continue

            # Create training example
            example = {
                'play_id': label['play_id'],
                'play_data': label['play_data'],
                'ground_truth': {
                    'outcome': label['outcome'],
                    'success': label['outcome'] == 'Success'
                },
                'tags': label['tags']
            }

            training_data['plays'].append(example)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"Exported {len(training_data['plays'])} labeled plays to {output_path}")

        return training_data

    def get_labeling_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on labeled data.

        Returns:
            Statistics dictionary
        """
        if not self.labels:
            return {
                'total_labels': 0,
                'outcomes': {},
                'avg_tags_per_play': 0
            }

        # Count outcomes
        outcomes = {}
        for label in self.labels:
            outcome = label['outcome']
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        # Count tags
        total_tags = sum(len(label.get('tags', [])) for label in self.labels)

        stats = {
            'total_labels': len(self.labels),
            'outcomes': outcomes,
            'avg_tags_per_play': total_tags / len(self.labels),
            'most_common_tags': self._get_most_common_tags(),
            'labeling_rate': self._calculate_labeling_rate()
        }

        return stats

    def _get_most_common_tags(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get most common tags."""
        tag_counts = {}

        for label in self.labels:
            for tag in label.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort by count
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

        return sorted_tags[:top_n]

    def _calculate_labeling_rate(self) -> Optional[float]:
        """Calculate labels per day."""
        if len(self.labels) < 2:
            return None

        # Parse timestamps
        timestamps = []
        for label in self.labels:
            try:
                ts = datetime.fromisoformat(label['timestamp'])
                timestamps.append(ts)
            except:
                continue

        if len(timestamps) < 2:
            return None

        # Calculate rate
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 86400  # days
        if time_span > 0:
            return len(timestamps) / time_span

        return None


def create_ui_label_payload(
    play_data: Dict[str, Any],
    insight: Dict[str, Any],
    al_manager: ActiveLearningManager
) -> Optional[Dict[str, Any]]:
    """
    Create payload for UI labeling interface.

    Args:
        play_data: Play data
        insight: Generated insight
        al_manager: Active learning manager

    Returns:
        UI payload if flagged, None otherwise
    """
    # Check if should flag
    xt_grid = insight.get('zones')
    if xt_grid is None or not al_manager.should_flag_for_review(xt_grid, insight):
        return None

    # Create review prompt
    prompt = al_manager.create_review_prompt(play_data, insight)

    # Add UI-specific fields
    payload = {
        **prompt,
        'ui_config': {
            'show_modal': True,
            'buttons': [
                {'id': 'success', 'label': 'Success', 'color': 'green'},
                {'id': 'fail', 'label': 'Fail', 'color': 'red'},
                {'id': 'intercepted', 'label': 'Intercepted', 'color': 'orange'}
            ],
            'tag_suggestions': [
                'LB late', 'CB overcommit', 'Safety blitz',
                'WR open', 'Pocket collapsed', 'Coverage bust'
            ]
        }
    }

    return payload
