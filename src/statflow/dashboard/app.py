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
    confidence_tier_breakdown,
    daily_metrics_trend,
    load_features_for_games,
    load_historical_predictions,
    load_prediction_outcomes,
    load_todays_games,
    rolling_performance,
    runs_scatter_data,
)
from statflow.models.data import FEATURE_COLS
from statflow.models.explanations import top_contributions


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


@st.cache_data(ttl=300)
def _cached_history() -> pd.DataFrame:
    return load_historical_predictions()


@st.cache_resource
def _cached_winner_model():
    """Load the latest XGBoost winner model from MLflow, once per app process.

    Returns (model, run_id) or (None, None) if MLflow isn't yet populated
    (e.g., first-ever run on a fresh bucket). The Today's-games explain
    panel just hides itself if this returns None.
    """
    try:
        from statflow.models.predict import load_latest_winner_model

        return load_latest_winner_model()
    except Exception:
        return None, None


def _explanations_for(game_pks: list[int]) -> dict[int, list[tuple[str, float]]]:
    """Return {game_pk: [(feature, contribution), ...]} for today's games.

    Returns {} — silently — if any prerequisite is missing (no model,
    empty features, stale features parquet missing new FEATURE_COLS, or
    the trained model expects different columns than what's on disk).
    The game cards just hide the "Why?" expander in that case.
    """
    if not game_pks:
        return {}
    model, _ = _cached_winner_model()
    if model is None:
        return {}
    features = load_features_for_games(game_pks)
    if features.empty:
        return {}
    # Guard against stale features parquet on R2 that predates newer
    # FEATURE_COLS additions (bullpen / IL columns, etc.).
    missing_cols = [c for c in FEATURE_COLS if c not in features.columns]
    if missing_cols:
        return {}
    ordered_pks = [pk for pk in game_pks if pk in features.index]
    if not ordered_pks:
        return {}
    try:
        X = features.loc[ordered_pks, FEATURE_COLS]
        contribs = top_contributions(model, X, FEATURE_COLS, top_n=5)
        return dict(zip(ordered_pks, contribs, strict=True))
    except Exception:
        return {}


tab_today, tab_perf, tab_history = st.tabs(
    ["Today's games", "Model performance", "Historical explorer"]
)

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
        explanations = _explanations_for(list(games["game_pk"]))
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

                # Per-game explanation — top 5 features that drove the prediction.
                # SHAP-equivalent values from XGBoost's pred_contribs, in log-odds space.
                pk = int(row["game_pk"])
                if pk in explanations:
                    with st.expander("Why?  Top 5 features driving this prediction"):
                        st.caption(
                            "Positive contributions pushed the model toward "
                            "**home win**; negative pushed it toward **away**. "
                            "Values are in log-odds — magnitudes are comparable across features."
                        )
                        contrib_df = pd.DataFrame(
                            explanations[pk], columns=["feature", "contribution"]
                        )
                        contrib_df["direction"] = contrib_df["contribution"].apply(
                            lambda v: "↑ home" if v > 0 else "↓ away"
                        )
                        contrib_df["contribution"] = contrib_df["contribution"].round(3)
                        st.dataframe(contrib_df, hide_index=True, use_container_width=False)

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
        # Performance trend over time — is the model getting better or worse?
        # -------------------------------------------------------------------
        st.subheader("Performance over time")
        st.caption(
            "Daily accuracy + log loss with a 7-day rolling average. Sparse "
            "days are individually noisy; the rolling line is what to watch "
            "for drift or improvement."
        )
        trend = daily_metrics_trend(outcomes)
        if len(trend) < 2:
            st.info(
                f"Only {len(trend)} day of data so far — the trend chart needs "
                "a few more days of daily-flow runs before it becomes meaningful."
            )
        else:
            trend_long = trend.melt(
                id_vars=["game_date"],
                value_vars=["accuracy", "accuracy_rolling"],
                var_name="series",
                value_name="value",
            )
            acc_chart = (
                alt.Chart(trend_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("game_date:T", title="Date"),
                    y=alt.Y("value:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(
                            domain=["accuracy", "accuracy_rolling"],
                            range=["#bbb", "#4c8bf5"],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=["game_date:T", alt.Tooltip("value:Q", format=".3f")],
                )
                .properties(height=220)
            )
            st.altair_chart(acc_chart, use_container_width=True)

        # -------------------------------------------------------------------
        # Confidence tiers — is the model MORE right when it's MORE confident?
        # -------------------------------------------------------------------
        st.subheader("Accuracy by confidence tier")
        st.caption(
            "How often the model is right, split by how confident it was. "
            "A useful model is meaningfully more accurate at 70%+ than at 50–55%."
        )
        tiers = confidence_tier_breakdown(outcomes)
        if not tiers.empty:
            display = tiers.copy()
            display["accuracy"] = (display["accuracy"] * 100).round(1).astype(str) + "%"
            display = display.rename(
                columns={"tier": "Confidence", "n": "Games", "accuracy": "Accuracy"}
            )
            st.dataframe(display, hide_index=True, use_container_width=False)

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


# ---------------------------------------------------------------------------
# Historical explorer
# ---------------------------------------------------------------------------
with tab_history:
    history = _cached_history()
    if history.empty:
        st.info(
            "No historical predictions yet — this tab populates as the daily "
            "flow runs and games complete."
        )
    else:
        st.caption(
            "Every prediction the model has ever made on a game that has "
            "since finished. Filter by date range or team, sort by biggest miss."
        )

        min_date = history["game_date"].min()
        max_date = history["game_date"].max()
        col_a, col_b = st.columns([2, 3])
        date_range = col_a.date_input(
            "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )
        all_teams = sorted(
            set(history["home_team_name"].dropna()) | set(history["away_team_name"].dropna())
        )
        selected_teams = col_b.multiselect("Teams (leave empty for all)", options=all_teams)

        filtered = history.copy()
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            filtered = filtered[(filtered["game_date"] >= start) & (filtered["game_date"] <= end)]
        if selected_teams:
            filtered = filtered[
                filtered["home_team_name"].isin(selected_teams)
                | filtered["away_team_name"].isin(selected_teams)
            ]

        st.caption(f"{len(filtered):,} matching games")

        display = filtered.copy()
        display["matchup"] = display["away_team_name"] + " @ " + display["home_team_name"]
        display["score"] = (
            display["away_score"].astype("Int64").astype(str)
            + "–"
            + display["home_score"].astype("Int64").astype(str)
        )
        display["home_win_prob"] = (display["predicted_home_win_prob"] * 100).round(1).astype(
            str
        ) + "%"
        display["actual"] = display["actual_home_win"].map({1: "home won", 0: "away won"})
        display["pred_runs"] = display["predicted_total_runs"].round(1)
        display["actual_runs"] = display["actual_total_runs"]
        display["runs_error"] = display["runs_abs_error"].round(2)
        display["hit"] = display["winner_correct"].map({True: "✓", False: "✗"})

        st.dataframe(
            display[
                [
                    "game_date",
                    "matchup",
                    "home_win_prob",
                    "actual",
                    "hit",
                    "pred_runs",
                    "actual_runs",
                    "runs_error",
                    "score",
                ]
            ].rename(
                columns={
                    "game_date": "Date",
                    "matchup": "Matchup (away @ home)",
                    "home_win_prob": "Home win prob",
                    "actual": "Actual",
                    "hit": "Correct?",
                    "pred_runs": "Predicted runs",
                    "actual_runs": "Actual runs",
                    "runs_error": "|error|",
                    "score": "Final",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
