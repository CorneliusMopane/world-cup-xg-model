"""
app.py

Interactive Streamlit dashboard for exploring the 2022 World Cup xG model.

Run locally with:
    streamlit run app.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc

DATA_PATH = Path(__file__).parent / "data" / "processed" / "shots.csv"
MODEL_PATH = Path(__file__).parent / "models" / "xg_model.pkl"

st.set_page_config(page_title="World Cup 2022 xG Explorer", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Collapse the detailed body_part categories into the same three
    # buckets (Head / Foot / Other) the model was trained on.
    df["body_part"] = df["body_part"].apply(
        lambda b: "Head" if b == "Head" else ("Foot" if "Foot" in str(b) else "Other")
    )
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def draw_pitch_half(ax):
    ax.add_patch(Rectangle((60, 0), 60, 80, fill=False, color="white", lw=1.5))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, color="white", lw=1.5))
    ax.add_patch(Rectangle((84, 18), 36, 44, fill=False, color="white", lw=1.5))
    ax.add_patch(Arc((108, 40), 20, 20, angle=0, theta1=128, theta2=232, color="white", lw=1.5))
    ax.add_patch(Rectangle((120, 36), 2, 8, fill=False, color="white", lw=2))
    ax.set_facecolor("#1e5631")
    ax.set_xlim(60, 122)
    ax.set_ylim(0, 80)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    st.title("⚽ 2022 FIFA World Cup — Expected Goals (xG) Explorer")
    st.caption(
        "A from-scratch xG model trained on real StatsBomb open-data event data, "
        "benchmarked against StatsBomb's own professional xG model "
        "([full write-up on GitHub](https://github.com/CorneliusMopane/world-cup-xg-model))."
    )

    df = load_data()
    model = load_model()

    features = ["distance_to_goal", "angle_to_goal", "body_part", "shot_type", "under_pressure"]
    df["our_xg"] = model.predict_proba(df[features])[:, 1]

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    teams = sorted(df["team"].dropna().unique())
    selected_teams = st.sidebar.multiselect("Team(s)", teams, default=[])

    filtered = df.copy()
    if selected_teams:
        filtered = filtered[filtered["team"].isin(selected_teams)]

    players = sorted(filtered["player"].dropna().unique())
    selected_player = st.sidebar.selectbox("Player (optional)", ["All players"] + players)
    if selected_player != "All players":
        filtered = filtered[filtered["player"] == selected_player]

    only_goals = st.sidebar.checkbox("Show goals only", value=False)
    if only_goals:
        filtered = filtered[filtered["is_goal"] == 1]

    # --- Top-line metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shots shown", len(filtered))
    col2.metric("Goals", int(filtered["is_goal"].sum()))
    col3.metric("Total xG", f"{filtered['our_xg'].sum():.2f}")
    if filtered["is_goal"].sum() > 0 and filtered["our_xg"].sum() > 0:
        overperf = filtered["is_goal"].sum() - filtered["our_xg"].sum()
        col4.metric("Goals vs. xG", f"{overperf:+.2f}",
                    help="Positive = outperforming expected goals (clinical finishing). "
                         "Negative = underperforming xG.")
    else:
        col4.metric("Goals vs. xG", "—")

    # --- Shot map ---
    st.subheader("Shot Map")
    if len(filtered) == 0:
        st.info("No shots match the current filters.")
    else:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        draw_pitch_half(ax)
        misses = filtered[filtered["is_goal"] == 0]
        goals = filtered[filtered["is_goal"] == 1]

        ax.scatter(misses["x"], misses["y"], s=misses["our_xg"] * 500 + 15,
                   c=misses["our_xg"], cmap="Reds", alpha=0.6, edgecolors="white",
                   linewidths=0.3, vmin=0, vmax=0.8)
        sc = ax.scatter(goals["x"], goals["y"], s=goals["our_xg"] * 500 + 40,
                        c=goals["our_xg"], cmap="Reds", alpha=0.95, edgecolors="black",
                        linewidths=1.0, marker="*", vmin=0, vmax=0.8)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="Predicted xG")
        st.pyplot(fig)
        st.caption("Stars = goals. Larger and redder = higher predicted xG.")

    # --- Player leaderboard ---
    st.subheader("xG Overperformance Leaderboard")
    st.caption("Players who scored more (or fewer) goals than their shot quality alone would predict — "
               "a proxy for finishing skill, minimum 3 shots.")
    leaderboard = (
        df.groupby(["player", "team"])
        .agg(shots=("is_goal", "count"), goals=("is_goal", "sum"), total_xg=("our_xg", "sum"))
        .reset_index()
    )
    leaderboard = leaderboard[leaderboard["shots"] >= 3]
    leaderboard["xg_overperformance"] = leaderboard["goals"] - leaderboard["total_xg"]
    leaderboard = leaderboard.sort_values("xg_overperformance", ascending=False)
    leaderboard["total_xg"] = leaderboard["total_xg"].round(2)
    leaderboard["xg_overperformance"] = leaderboard["xg_overperformance"].round(2)

    tab1, tab2 = st.tabs(["Overperforming xG", "Underperforming xG"])
    with tab1:
        st.dataframe(leaderboard.head(10), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(leaderboard.tail(10).sort_values("xg_overperformance"),
                     use_container_width=True, hide_index=True)

    # --- Raw data ---
    with st.expander("View filtered shot data"):
        st.dataframe(
            filtered[["team", "player", "distance_to_goal", "angle_to_goal",
                      "body_part", "under_pressure", "our_xg", "statsbomb_xg", "is_goal"]]
            .round(3),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
