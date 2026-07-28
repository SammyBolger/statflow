"""StatFlow dashboard.

Local dev:
    uv run streamlit run src/statflow/dashboard/app.py

Hosted (Streamlit Community Cloud): same file. When R2 credentials are
provided via Streamlit's secrets store, the app pulls the latest silver/
gold state from R2 into a local temp dir on cold-start, so the same
loaders work in both environments.
"""

from __future__ import annotations

import os
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from statflow.dashboard.data import (
    baseline_comparison,
    calibration_bins,
    load_prediction_outcomes,
    load_todays_games,
    rolling_performance,
    runs_scatter_data,
)


@st.cache_resource
def _bootstrap_from_r2() -> str:
    """On cold-start, if R2 secrets are configured, pull silver/gold from R2.

    Runs once per app process (cache_resource semantics). No-op locally when
    R2 isn't set — the loaders then read data/ off local disk, which is how
    local dev works. Any error is caught + surfaced as a warning so the
    dashboard still boots (just with empty data).
    """
    try:
        has_r2 = "R2_ACCOUNT_ID" in st.secrets
    except Exception:
        # No secrets.toml (typical local dev) — st.secrets access can raise.
        has_r2 = False
    if not has_r2:
        return "R2 not configured — using local data/"

    for key in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])

    try:
        from statflow.storage.r2 import pull

        n = pull()
        return f"pulled {n} file(s) from R2"
    except Exception as exc:
        st.warning(f"R2 pull failed — showing whatever's cached: {exc}")
        return f"R2 pull failed: {exc}"


_bootstrap_from_r2()

# One-line help strings so every metric has a hoverable "?" explaining it.
# No hardcoded baseline numbers — those are computed live in the
# model-vs-baseline chart below, so we point users there instead of embedding
# a number that could drift out of sync as more data accumulates.
METRIC_HELP = {
    "games": "Number of completed Final games in the rolling window.",
    "accuracy": (
        "% of games where the model correctly picked the winner "
        "(predicted probability > 50% for the team that actually won). "
        "Compare vs the 'always pick home' baseline in the chart below."
    ),
    "log_loss": (
        "Penalizes wrong-and-confident predictions much more than "
        "wrong-and-hedgy ones. Lower is better. Compare vs baseline "
        "in the chart below — beating it means the model is adding signal."
    ),
    "brier": (
        "Mean squared error on the predicted probabilities. Lower is "
        "better. Compare vs baseline in the chart below."
    ),
    "mae": (
        "Average absolute error in predicted total runs (in runs). "
        "Lower is better. Compare vs the mean-runs baseline in the "
        "chart below."
    ),
    "rmse": (
        "Root mean squared error on total runs. Similar to MAE but "
        "penalizes big misses more heavily."
    ),
}

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
                cols = st.columns([4, 3, 1, 1])

                # Matchup with MLB team logos (public CDN, keyed by team_id).
                away_logo = f"https://www.mlbstatic.com/team-logos/{row['away_team_id']}.svg"
                home_logo = f"https://www.mlbstatic.com/team-logos/{row['home_team_id']}.svg"
                matchup = (
                    f'<img src="{away_logo}" width="30" '
                    f'style="vertical-align:middle;margin-right:6px">'
                    f"<b>{row['away_team_name']}</b>"
                    f" &nbsp;@&nbsp; "
                    f'<img src="{home_logo}" width="30" '
                    f'style="vertical-align:middle;margin-right:6px">'
                    f"<b>{row['home_team_name']}</b>"
                )
                cols[0].markdown(matchup, unsafe_allow_html=True)

                away_sp = row.get("away_probable_pitcher_name") or "(TBD)"
                home_sp = row.get("home_probable_pitcher_name") or "(TBD)"
                cols[1].markdown(f"SP: {away_sp} vs {home_sp}")

                # Show whichever team the model favors, by name, with its own
                # win probability — much clearer than "Home win 68%".
                prob = row.get("predicted_home_win_prob")
                if pd.notna(prob):
                    if prob >= 0.5:
                        favored = row["home_team_name"]
                        favored_prob = prob
                    else:
                        favored = row["away_team_name"]
                        favored_prob = 1 - prob
                    cols[2].metric(favored, f"{favored_prob * 100:.0f}%")

                total = row.get("predicted_total_runs")
                if pd.notna(total):
                    cols[3].metric("Total runs", f"{total:.1f}")

                if row.get("status") == "Final":
                    if row["home_win"]:
                        winner, w_score, l_score = (
                            row["home_team_name"],
                            row["home_score"],
                            row["away_score"],
                        )
                    else:
                        winner, w_score, l_score = (
                            row["away_team_name"],
                            row["away_score"],
                            row["home_score"],
                        )
                    cols[0].caption(f"Final: **{winner}** won {w_score}–{l_score}")

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
        # -------------------------------------------------------------------
        # Rolling headline metrics (with hoverable "?" tooltips)
        # -------------------------------------------------------------------
        st.subheader("Rolling performance")
        window = st.selectbox("Window (days)", [7, 30, 90], index=1)
        metrics = rolling_performance(outcomes, window)
        if not metrics:
            st.warning("Not enough data for this window.")
        else:
            c = st.columns(6)
            c[0].metric("Games", metrics["n_games"], help=METRIC_HELP["games"])
            c[1].metric(
                "Accuracy", f"{metrics['accuracy'] * 100:.1f}%", help=METRIC_HELP["accuracy"]
            )
            c[2].metric("Log loss", f"{metrics['log_loss']:.4f}", help=METRIC_HELP["log_loss"])
            c[3].metric("Brier", f"{metrics['brier']:.4f}", help=METRIC_HELP["brier"])
            c[4].metric("MAE runs", f"{metrics['mae']:.2f}", help=METRIC_HELP["mae"])
            c[5].metric("RMSE runs", f"{metrics['rmse']:.2f}", help=METRIC_HELP["rmse"])

        # -------------------------------------------------------------------
        # Model vs baseline bar chart — the "is this model earning its keep?"
        # -------------------------------------------------------------------
        st.subheader("Model vs baseline")
        st.caption(
            "Naive baselines: always predict home wins (for the winner target) "
            "and always predict the mean total runs (for the runs target). If "
            "the blue bar isn't beating the gray bar on the 'lower is better' "
            "metrics, the model isn't adding signal yet."
        )
        compare = baseline_comparison(outcomes)
        if not compare.empty:
            long = compare.melt(
                id_vars=["metric"],
                value_vars=["model", "baseline"],
                var_name="source",
                value_name="value",
            )
            chart = (
                alt.Chart(long)
                .mark_bar()
                .encode(
                    x=alt.X("source:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("value:Q", title=None),
                    color=alt.Color(
                        "source:N",
                        scale=alt.Scale(domain=["model", "baseline"], range=["#4c8bf5", "#888"]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=["metric", "source", alt.Tooltip("value:Q", format=".4f")],
                )
                .properties(width=120, height=180)
                .facet(
                    column=alt.Column("metric:N", title=None, header=alt.Header(labelFontSize=13))
                )
                .resolve_scale(y="independent")
            )
            st.altair_chart(chart, use_container_width=False)

        # -------------------------------------------------------------------
        # Predicted vs actual runs — an interpretable single-chart view.
        # -------------------------------------------------------------------
        st.subheader("Predicted vs actual total runs")
        st.caption(
            "Each dot is one completed game. Perfect predictions lie on the "
            "dashed diagonal. Dots consistently below the diagonal = model "
            "over-predicting; above = under-predicting."
        )
        scatter = runs_scatter_data(outcomes)
        if not scatter.empty:
            axis_max = (
                float(
                    max(scatter["predicted_total_runs"].max(), scatter["actual_total_runs"].max())
                )
                + 1
            )
            points = (
                alt.Chart(scatter)
                .mark_circle(size=60, opacity=0.6, color="#4c8bf5")
                .encode(
                    x=alt.X(
                        "predicted_total_runs:Q",
                        title="Predicted total runs",
                        scale=alt.Scale(domain=[0, axis_max]),
                    ),
                    y=alt.Y(
                        "actual_total_runs:Q",
                        title="Actual total runs",
                        scale=alt.Scale(domain=[0, axis_max]),
                    ),
                    tooltip=[
                        "game_pk",
                        alt.Tooltip("predicted_total_runs:Q", format=".2f"),
                        "actual_total_runs:Q",
                    ],
                )
            )
            diag = (
                alt.Chart(pd.DataFrame({"x": [0, axis_max], "y": [0, axis_max]}))
                .mark_line(strokeDash=[4, 4], color="gray")
                .encode(x="x:Q", y="y:Q")
            )
            st.altair_chart(points + diag, use_container_width=True)

        # -------------------------------------------------------------------
        # Calibration plot — kept, but flagged as needing more data.
        # -------------------------------------------------------------------
        st.subheader("Calibration (win probability)")
        n_outcomes = len(outcomes)
        if n_outcomes >= 200:
            maturity = f"Sample: **{n_outcomes:,} completed games** — enough for meaningful bins."
        else:
            maturity = (
                f"Sample: **{n_outcomes:,} completed games** — becomes meaningful around ~200."
            )
        st.caption(
            "When the model says X%, does the team actually win X% of the time? "
            "Each dot is a probability bin; well-calibrated points lie on the "
            f"y=x diagonal. {maturity}"
        )
        cal = calibration_bins(outcomes)
        if not cal.empty:
            base = alt.Chart(cal).encode(
                x=alt.X(
                    "mean_predicted:Q",
                    title="Predicted home-win probability",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "mean_actual:Q",
                    title="Actual home-win rate",
                    scale=alt.Scale(domain=[0, 1]),
                ),
            )
            points = base.mark_circle(color="#4c8bf5").encode(
                size=alt.Size("n:Q", title="Games in bin"),
                tooltip=["mean_predicted", "mean_actual", "n"],
            )
            diag = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(strokeDash=[4, 4], color="gray")
                .encode(x="x:Q", y="y:Q")
            )
            st.altair_chart(points + diag, use_container_width=True)
