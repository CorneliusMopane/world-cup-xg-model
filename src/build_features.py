"""
build_features.py

Extracts every shot from the cached World Cup 2022 event files and
turns each one into a row of features suitable for modeling:

    distance_to_goal   - straight-line distance from the shot location
                          to the center of the goal
    angle_to_goal      - the angle (in degrees) subtended by the goal
                          mouth from the shot location; a shot from
                          right in front of goal has a wide angle,
                          a shot from the byline has a narrow one
    body_part          - Head / Foot / Other
    is_open_play       - was this from open play, or a set piece?
    under_pressure     - was a defender closing the shooter down?
    is_goal            - the target we want to predict (1 or 0)

Pitch dimensions (StatsBomb convention): 120 (length) x 80 (width).
The goal sits on the line x = 120, centered at y = 40, and is
7.32 units wide (the real-world 7.32m goal width, kept in the same
units as the pitch coordinates).
"""
import json
import math
from pathlib import Path
import pandas as pd

EVENTS_DIR = Path(__file__).parent.parent / "data" / "raw" / "events"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "shots.csv"

GOAL_X = 120
GOAL_Y_CENTER = 40
GOAL_HALF_WIDTH = 7.32 / 2  # 3.66 either side of center


def compute_distance_and_angle(x: float, y: float) -> tuple[float, float]:
    """Given a shot location, compute distance to goal center and the
    angle (in degrees) the goal mouth subtends from that point."""
    dx = GOAL_X - x
    dy_left = (GOAL_Y_CENTER - GOAL_HALF_WIDTH) - y
    dy_right = (GOAL_Y_CENTER + GOAL_HALF_WIDTH) - y

    distance = math.hypot(dx, GOAL_Y_CENTER - y)

    # Angle between the two lines from the shot location to each goalpost.
    angle_left = math.atan2(dy_left, dx)
    angle_right = math.atan2(dy_right, dx)
    angle = abs(math.degrees(angle_left - angle_right))

    return distance, angle


def extract_shots_from_match(match_id: int) -> list[dict]:
    with open(EVENTS_DIR / f"{match_id}.json") as f:
        events = json.load(f)

    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue

        shot = e["shot"]

        # Penalties are a different animal (near-certain, very short
        # distance) so we exclude them to keep the model about open
        # play / set-piece shooting skill, not penalty-taking.
        if shot.get("type", {}).get("name") == "Penalty":
            continue

        x, y = e["location"]
        distance, angle = compute_distance_and_angle(x, y)

        rows.append({
            "match_id": match_id,
            "team": e.get("team", {}).get("name"),
            "player": e.get("player", {}).get("name"),
            "x": x,
            "y": y,
            "distance_to_goal": distance,
            "angle_to_goal": angle,
            "body_part": shot.get("body_part", {}).get("name", "Unknown"),
            "shot_type": shot.get("type", {}).get("name", "Unknown"),
            "technique": shot.get("technique", {}).get("name", "Unknown"),
            "under_pressure": bool(e.get("under_pressure", False)),
            "statsbomb_xg": shot.get("statsbomb_xg"),
            "is_goal": 1 if shot.get("outcome", {}).get("name") == "Goal" else 0,
        })

    return rows


def main():
    match_files = sorted(EVENTS_DIR.glob("*.json"))
    all_rows = []
    for path in match_files:
        match_id = int(path.stem)
        all_rows.extend(extract_shots_from_match(match_id))

    df = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Extracted {len(df)} shots from {len(match_files)} matches")
    print(f"Overall goal conversion rate: {df['is_goal'].mean():.1%}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
