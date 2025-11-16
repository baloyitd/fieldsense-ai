"""
Data ingestion module for FieldSense AI v3.0
Handles loading of STATSBomb JSON, Wyscout CSV, and Custom CSV formats
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


def load_statsbomb_json(file_path: str) -> pd.DataFrame:
    """
    Load STATSBomb JSON events data and parse to DataFrame.

    Args:
        file_path: Path to STATSBomb JSON file

    Returns:
        DataFrame with columns: match_id, event_id, x, y, velocity, direction,
        acceleration, possession, role
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    events = []
    for event in data:
        # Parse STATSBomb event structure
        event_dict = {
            'match_id': event.get('match_id', 0),
            'event_id': event.get('id', 0),
            'x': event.get('location', [0, 0])[0] if event.get('location') else 0.0,
            'y': event.get('location', [0, 0])[1] if event.get('location') else 0.0,
            'velocity': event.get('velocity', 0.0),
            'direction': event.get('direction', 0.0),
            'acceleration': event.get('acceleration', 0.0),
            'possession': event.get('possession', False),
            'role': event.get('position', {}).get('name', 'Unknown') if isinstance(event.get('position'), dict) else 'Unknown',
            'player_id': event.get('player', {}).get('id', 0) if isinstance(event.get('player'), dict) else 0,
            'timestamp': event.get('timestamp', 0.0)
        }
        events.append(event_dict)

    df = pd.DataFrame(events)
    return df


def load_wyscout_csv(file_path: str) -> pd.DataFrame:
    """
    Load Wyscout CSV events data.

    Args:
        file_path: Path to Wyscout events.csv file

    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(file_path)

    # Wyscout uses positions array, parse if present
    if 'positions' in df.columns:
        # Extract x, y from positions if it's a string representation
        def parse_positions(pos_str):
            try:
                if pd.isna(pos_str):
                    return 0.0, 0.0
                # Assuming format like "[{'x': 50, 'y': 50}]"
                import ast
                positions = ast.literal_eval(pos_str) if isinstance(pos_str, str) else pos_str
                if positions and len(positions) > 0:
                    return positions[0].get('x', 0.0), positions[0].get('y', 0.0)
            except:
                pass
            return 0.0, 0.0

        df[['x', 'y']] = df['positions'].apply(lambda x: pd.Series(parse_positions(x)))

    # Standardize column names
    column_mapping = {
        'eventId': 'event_id',
        'matchId': 'match_id',
        'playerId': 'player_id',
        'teamId': 'team_id'
    }
    df = df.rename(columns=column_mapping)

    # Add missing columns with defaults
    if 'velocity' not in df.columns:
        df['velocity'] = 0.0
    if 'direction' not in df.columns:
        df['direction'] = 0.0
    if 'acceleration' not in df.columns:
        df['acceleration'] = 0.0
    if 'possession' not in df.columns:
        df['possession'] = False
    if 'role' not in df.columns:
        df['role'] = 'Unknown'
    if 'timestamp' not in df.columns:
        df['timestamp'] = 0.0

    return df


def load_custom_csv(file_path: str) -> pd.DataFrame:
    """
    Load Custom CSV tracking data (NFL-style).

    Args:
        file_path: Path to tracking_gameId.csv file

    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(file_path)

    # Map NFL-style columns to our standard
    column_mapping = {
        'gameId': 'match_id',
        'playId': 'event_id',
        's': 'velocity',
        'dir': 'direction',
        'a': 'acceleration',
        'nflId': 'player_id',
        'time': 'timestamp'
    }

    # Rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    # Ensure required columns exist
    if 'x' not in df.columns:
        df['x'] = 0.0
    if 'y' not in df.columns:
        df['y'] = 0.0
    if 'velocity' not in df.columns:
        df['velocity'] = 0.0
    if 'direction' not in df.columns:
        df['direction'] = 0.0
    if 'acceleration' not in df.columns:
        df['acceleration'] = 0.0
    if 'possession' not in df.columns:
        df['possession'] = False
    if 'role' not in df.columns:
        df['role'] = df.get('position', 'Unknown')
    if 'timestamp' not in df.columns:
        df['timestamp'] = 0.0

    return df


def load_data(file_path: str) -> pd.DataFrame:
    """
    Auto-detect format and load data file.

    Args:
        file_path: Path to data file

    Returns:
        DataFrame with standardized columns
    """
    path = Path(file_path)

    if path.suffix.lower() == '.json':
        return load_statsbomb_json(file_path)
    elif path.suffix.lower() == '.csv':
        # Try to detect CSV type by columns
        df_sample = pd.read_csv(file_path, nrows=1)

        if 'positions' in df_sample.columns or 'eventId' in df_sample.columns:
            return load_wyscout_csv(file_path)
        elif 'gameId' in df_sample.columns or 'playId' in df_sample.columns:
            return load_custom_csv(file_path)
        else:
            # Default to custom format
            return load_custom_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
