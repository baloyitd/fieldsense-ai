"""
Data normalization module for FieldSense AI v3.0
Converts all input formats to unified schema
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.cluster import KMeans
import json


# Field dimensions: 105m x 68m (standard soccer pitch)
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def scale_coordinates(x: float, y: float, source_length: float = 120.0, source_width: float = 80.0) -> tuple:
    """
    Scale coordinates to standard field dimensions (105m x 68m).

    Args:
        x: X coordinate
        y: Y coordinate
        source_length: Source field length
        source_width: Source field width

    Returns:
        Tuple of (scaled_x, scaled_y)
    """
    scaled_x = (x / source_length) * FIELD_LENGTH
    scaled_y = (y / source_width) * FIELD_WIDTH
    return scaled_x, scaled_y


def auto_detect_roles(df: pd.DataFrame, n_clusters: int = 3) -> pd.Series:
    """
    Auto-detect player roles using position clustering (KMeans).

    Args:
        df: DataFrame with x, y coordinates
        n_clusters: Number of clusters (default 3: defensive, midfield, offensive)

    Returns:
        Series with predicted roles
    """
    if len(df) < n_clusters:
        return pd.Series(['Unknown'] * len(df))

    # Use average positions for clustering
    features = df[['x', 'y']].fillna(0).values

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)

        # Map clusters to roles based on average y position
        cluster_centers = kmeans.cluster_centers_
        cluster_y_means = cluster_centers[:, 1]
        sorted_clusters = np.argsort(cluster_y_means)

        role_mapping = {}
        role_names = ['Defender', 'Midfielder', 'Forward']
        for i, cluster_idx in enumerate(sorted_clusters):
            role_mapping[cluster_idx] = role_names[i] if i < len(role_names) else 'Unknown'

        roles = pd.Series([role_mapping[c] for c in clusters])
        return roles
    except Exception as e:
        print(f"Warning: Role detection failed: {e}")
        return pd.Series(['Unknown'] * len(df))


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    # Impute velocity, direction, acceleration with 0
    for col in ['velocity', 'direction', 'acceleration']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Impute coordinates with field center if missing
    if 'x' in df.columns:
        df['x'] = df['x'].fillna(FIELD_LENGTH / 2.0)
    if 'y' in df.columns:
        df['y'] = df['y'].fillna(FIELD_WIDTH / 2.0)

    # Impute boolean possession with False
    if 'possession' in df.columns:
        df['possession'] = df['possession'].fillna(False)

    # Impute role with Unknown
    if 'role' in df.columns:
        df['role'] = df['role'].fillna('Unknown')

    return df


def normalize_dataframe(df: pd.DataFrame, auto_roles: bool = True) -> List[Dict[str, Any]]:
    """
    Normalize DataFrame to unified schema.

    Args:
        df: Input DataFrame from ingest module
        auto_roles: Whether to auto-detect roles if missing

    Returns:
        List of frame dictionaries in unified schema
    """
    # Impute missing values
    df = impute_missing_values(df)

    # Scale coordinates if needed (assume input might be in different scales)
    # Check if coordinates are already in 0-105, 0-68 range
    max_x = df['x'].max()
    max_y = df['y'].max()

    if max_x > 110 or max_y > 75:
        # Likely need scaling (e.g., from 120x80 or yards)
        df['x'], df['y'] = zip(*df.apply(lambda row: scale_coordinates(row['x'], row['y']), axis=1))

    # Clip to field boundaries
    df['x'] = df['x'].clip(0, FIELD_LENGTH)
    df['y'] = df['y'].clip(0, FIELD_WIDTH)

    # Auto-detect roles if needed
    if auto_roles and (df['role'] == 'Unknown').any():
        detected_roles = auto_detect_roles(df)
        df.loc[df['role'] == 'Unknown', 'role'] = detected_roles[df['role'] == 'Unknown']

    # Group by timestamp/frame to create frame structure
    # Use match_id and timestamp if available, otherwise create synthetic frames
    if 'timestamp' in df.columns:
        df['frame_time'] = df['timestamp']
    else:
        # Create synthetic time based on row index
        df['frame_time'] = df.index * 0.1  # 100ms intervals

    # Group by match and time
    frames = []
    if 'match_id' in df.columns:
        grouped = df.groupby(['match_id', 'frame_time'])
    else:
        grouped = df.groupby('frame_time')

    for group_key, group_df in grouped:
        frame = {
            'time': group_df['frame_time'].iloc[0],
            'players': [],
            'ball': {'x': FIELD_LENGTH / 2.0, 'y': FIELD_WIDTH / 2.0}
        }

        for idx, row in group_df.iterrows():
            player = {
                'id': str(row.get('player_id', idx)),
                'x': float(row['x']),
                'y': float(row['y']),
                'vel': float(row['velocity']),
                'dir': float(row['direction']),
                'acc': float(row['acceleration']),
                'role': str(row['role']),
                'possession': bool(row['possession'])
            }
            frame['players'].append(player)

            # If player has possession, update ball position
            if player['possession']:
                frame['ball']['x'] = player['x']
                frame['ball']['y'] = player['y']

        frames.append(frame)

    return frames


def normalize_to_json(df: pd.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalize DataFrame and optionally save to JSON.

    Args:
        df: Input DataFrame
        output_path: Optional path to save JSON output

    Returns:
        Dictionary with unified schema: {"frames": [...]}
    """
    frames = normalize_dataframe(df)
    output = {'frames': frames}

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    return output


def load_and_normalize(file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load data file and normalize to unified schema.

    Args:
        file_path: Path to input data file
        output_path: Optional path to save normalized JSON

    Returns:
        Dictionary with unified schema
    """
    from .ingest import load_data

    df = load_data(file_path)
    return normalize_to_json(df, output_path)
