"""
visualize.py

Produces two figures for the README / portfolio:

  1. shot_map.png        - every shot in the tournament, plotted on a
                            pitch, colored by our model's predicted xG
                            and sized/marked by whether it was a goal.
  2. calibration_plot.png - reliability curve comparing our model's
                            predicted probabilities to actual outcomes,
                            i.e. "when we say 30% chance, do goals
                            actually happen ~30% of the time?"
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc
from sklearn.calibration import calibration_curve

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "shots.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "xg_model.pkl"
FIG_DIR = Path(__file__).parent.parent / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_pitch_half(ax):
    """Draw the attacking half of a football pitch (StatsBomb 120x80 coords)."""
    ax.add_patch(Rectangle((60, 0), 60, 80, fill=False, color="white", lw=1.5))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, color="white", lw=1.5))  # 6-yard box
    ax.add_patch(Rectangle((84, 18), 36, 44, fill=False, color="white", lw=1.5))   # penalty box
    ax.add_patch(Arc((108, 40), 20, 20, angle=0, theta1=128, theta2=232, color="white", lw=1.5))
    ax.add_patch(Rectangle((120, 36), 2, 8, fill=False, color="white", lw=2))       # goal
    ax.set_facecolor("#1e5631")
    ax.set_xlim(60, 122)
    ax.set_ylim(0, 80)
    ax.set_xticks([])
    ax.set_yticks([])


def make_shot_map(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_pitch_half(ax)

    misses = df[df["is_goal"] == 0]
    goals = df[df["is_goal"] == 1]

    ax.scatter(misses["x"], misses["y"], s=misses["our_xg"] * 500 + 10,
               c=misses["our_xg"], cmap="Reds", alpha=0.55, edgecolors="white",
               linewidths=0.3, vmin=0, vmax=0.8, label="No goal")
    sc = ax.scatter(goals["x"], goals["y"], s=goals["our_xg"] * 500 + 30,
                     c=goals["our_xg"], cmap="Reds", alpha=0.95, edgecolors="black",
                     linewidths=1.0, marker="*", vmin=0, vmax=0.8, label="Goal")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Our model's predicted xG")
    ax.set_title("2022 FIFA World Cup — Every Shot, Colored by Predicted xG\n"
                  "(stars = actual goals, size/color = higher xG)", color="black", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shot_map.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'shot_map.png'}")


def make_calibration_plot(df):
    prob_true, prob_pred = calibration_curve(df["is_goal"], df["our_xg"], n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", color="crimson", label="Our xG model")
    ax.set_xlabel("Mean predicted xG (per bin)")
    ax.set_ylabel("Actual goal rate (per bin)")
    ax.set_title("Model Calibration: Predicted xG vs. Actual Goal Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_plot.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'calibration_plot.png'}")


def main():
    df = pd.read_csv(DATA_PATH)
    df["body_part"] = df["body_part"].apply(
        lambda b: "Head" if b == "Head" else ("Foot" if "Foot" in str(b) else "Other")
    )
    pipe = joblib.load(MODEL_PATH)
    features = ["distance_to_goal", "angle_to_goal", "body_part", "shot_type", "under_pressure"]
    df["our_xg"] = pipe.predict_proba(df[features])[:, 1]

    make_shot_map(df)
    make_calibration_plot(df)


if __name__ == "__main__":
    main()
