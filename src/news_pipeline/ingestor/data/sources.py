from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple


def load_source_bias_data() -> dict:
    """Load source bias data from the JSON file."""
    json_path = Path(__file__).parent / "source_bias.json"
    with open(json_path, "r") as f:
        return {
            source["Source"]: (source["Bias (X-axis)"], source["News Type (Y-axis)"])
            for source in json.load(f)
        }


# Load source bias data once when module is imported
SOURCE_COORDINATES = load_source_bias_data()


def get_source_coordinates(source: str) -> Optional[Tuple[float, float]]:
    """
    Get the coordinates (bias, news type) for a given news source.

    Args:
        source: The name of the news source

    Returns:
        Tuple of (x, y) coordinates where:
            x: Bias coordinate (-100 to 100, left to right)
            y: News type coordinate (-100 to 100, opinion to fact)
        Returns None if source is not found
    """
    # Try exact match first
    if source in SOURCE_COORDINATES:
        return SOURCE_COORDINATES[source]

    # Try case-insensitive match
    source_lower = source.lower()
    for known_source, coords in SOURCE_COORDINATES.items():
        if known_source.lower() == source_lower:
            return coords

    # Return None instead of raising ValueError to allow fallback
    return None
