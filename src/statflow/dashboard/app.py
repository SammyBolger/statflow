"""StatFlow dashboard.

Run with:
    uv run streamlit run src/statflow/dashboard/app.py
"""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from statflow.dashboard.data import (
    calibration_bins,
    load_prediction_outcomes,
    load_todays_games,
    rolling_performance,
)

st.set_page_config(page_title="StatFlow", layout="wide")
st.title("StatFlow — MLB game predictions")


@st.cache_data(ttl=300)
def _cached_todays(d: date) -> pd.DataFrame:
    return load_todays_games(d)


@st.cache_data(ttl=300)
def _cached_outcomes() -> pd.DataFrame:
    return load_prediction_outcomes()


tab_today, tab_perf = st.tabs(["Today's games", "Model performance"])

# ---------------------------------------------------------------------------
# Today's games
# ---------------------------------------------------------------------------
with tab_today:
    target = st.date_input("Date", value=date.today())
    games = _cached_todays(target)

    if games.empty:
        st.info("No games on this date, or silver hasn't been built yet.")
    else:
        st.caption(f"{len(games)} games")
        for _, row in games.iterrows():
            with st.container(border=True):
                cols = st.columns([3, 3, 1, 1])
                matchup = f"**{row['away_team_name']}** @ **{row['home_team_name']}**"
                cols[0].markdown(matchup)
                away_sp = row.get("away_probable_pitcher_name") or "(TBD)"
                home_sp = row.get("home_probable_pitcher_name") or "(TBD)"
                cols[1].markdown(f"SP: {away_sp} vs {home_sp}")

                prob = row.get("predicted_home_win_prob")
                if pd.notna(prob):
                    cols[2].metric("Home win", f"{prob * 100:.0f}%")
                total = row.get("predicted_total_runs")
                if pd.notna(total):
                    cols[3].metric("Total runs", f"{total:.1f}")

                if row.get("status") == "Final":
                    cols[0].caption(
                        f"Final: {row['away_score']}–{row['home_score']} "
                        f"({'home won' if row['home_win'] else 'away won'})"
                    )

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
with tab_perf:
    outcomes = _cached_outcomes()
    if outcomes.empty:
        st.info(
            "No prediction outcomes yet — need predictions on games that have finished. "
            "Run `python -m statflow.flows` daily to accumulate them."
        )
    else:
        st.subheader("Rolling performance")
        window = st.selectbox("Window (days)", [7, 30, 90], index=1)
        metrics = rolling_performance(outcomes, window)
        if not metrics:
            st.warning("Not enough data for this window.")
        else:
            c = st.columns(6)
            c[0].metric("Games", metrics["n_games"])
            c[1].metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            c[2].metric("Log loss", f"{metrics['log_loss']:.4f}")
            c[3].metric("Brier", f"{metrics['brier']:.4f}")
            c[4].metric("MAE runs", f"{metrics['mae']:.2f}")
            c[5].metric("RMSE runs", f"{metrics['rmse']:.2f}")

        st.subheader("Calibration")
        st.caption(
            "Each dot is a probability bin. A well-calibrated model lies on the "
            "y=x diagonal (predictions match actual win rates)."
        )
        cal = calibration_bins(outcomes)
        if not cal.empty:
            base = alt.Chart(cal).encode(
                x=alt.X("mean_predicted:Q", title="Predicted home-win probability"),
                y=alt.Y("mean_actual:Q", title="Actual home-win rate"),
            )
            points = base.mark_circle().encode(
                size=alt.Size("n:Q", title="Games in bin"),
                tooltip=["mean_predicted", "mean_actual", "n"],
            )
            diag = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(strokeDash=[3, 3], color="gray")
                .encode(x="x", y="y")
            )
            st.altair_chart(points + diag, use_container_width=True)
