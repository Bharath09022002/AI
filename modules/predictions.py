"""
Predictive module for the AI Coach — J.A.R.V.I.S. Edition.
Predicts streak break risk, low motivation risk, habit inconsistency,
projected level-up timelines, and optimal scheduling windows.
Uses rolling averages, trend slopes, exponential smoothing,
and recent activity patterns.
"""

import numpy as np
import pandas as pd
from models.schemas import TrackerEntry
from modules.analyzer import _to_dataframe


def predict_risks(
    tracker_data: list[TrackerEntry],
    detected_patterns: list[dict],
) -> dict:
    """
    Generate predictive risk scores based on tracker data and
    detected patterns.

    Returns:
        dict with keys:
            - streak_break_risk: float (0.0 – 1.0)
            - low_motivation_risk: float (0.0 – 1.0)
            - inconsistency_index: float (0.0 – 1.0)
            - risk_summary: str (J.A.R.V.I.S.-style human-readable)
            - at_risk_habits: list[str]
            - projected_streak_days: int
            - optimal_focus_time: str
            - system_health_score: float (0 – 100)
            - trend_direction: str
            - recovery_probability: float
    """
    df = _to_dataframe(tracker_data)

    if df.empty or len(df) < 3:
        return _empty_predictions()

    streak_risk = _calculate_streak_break_risk(df, detected_patterns)
    motivation_risk = _calculate_motivation_risk(df, detected_patterns)
    inconsistency = _calculate_inconsistency_index(df)
    at_risk = _identify_at_risk_habits(df)

    # ── ADVANCED: Projected streak continuation ──────────────────────────
    projected_streak = _project_streak(df)

    # ── ADVANCED: Optimal focus time estimation ──────────────────────────
    optimal_focus = _estimate_optimal_focus(df)

    # ── ADVANCED: System health score (0 – 100) ─────────────────────────
    health = _system_health_score(df, streak_risk, motivation_risk, inconsistency)

    # ── ADVANCED: Trend direction ────────────────────────────────────────
    trend = _overall_trend_direction(df)

    # ── ADVANCED: Recovery probability ───────────────────────────────────
    recovery_prob = _recovery_probability(df, streak_risk)

    # Build summary
    risks = []
    if streak_risk > 0.7:
        risks.append("⚠️ Critical streak disruption risk")
    elif streak_risk > 0.4:
        risks.append("⚡ Moderate streak break risk")

    if motivation_risk > 0.7:
        risks.append("⚠️ Motivation levels critically low")
    elif motivation_risk > 0.4:
        risks.append("⚡ Motivation may be wavering")

    if inconsistency > 0.6:
        risks.append("⚠️ Highly erratic behavior patterns")

    if health < 40:
        risks.append(f"🔴 System health critical ({health:.0f}/100)")
    elif health < 60:
        risks.append(f"🟡 System health suboptimal ({health:.0f}/100)")

    summary = "; ".join(risks) if risks else f"Systems nominal, Sir. Health score: {health:.0f}/100."

    return {
        "streak_break_risk": round(streak_risk, 2),
        "low_motivation_risk": round(motivation_risk, 2),
        "inconsistency_index": round(inconsistency, 2),
        "risk_summary": summary,
        "at_risk_habits": at_risk,
        "projected_streak_days": projected_streak,
        "optimal_focus_time": optimal_focus,
        "system_health_score": round(health, 1),
        "trend_direction": trend,
        "recovery_probability": round(recovery_prob, 2),
    }


def _empty_predictions() -> dict:
    return {
        "streak_break_risk": 0.0,
        "low_motivation_risk": 0.0,
        "inconsistency_index": 0.0,
        "risk_summary": "Insufficient data for predictive analysis, Sir.",
        "at_risk_habits": [],
        "projected_streak_days": 0,
        "optimal_focus_time": "N/A",
        "system_health_score": 50.0,
        "trend_direction": "insufficient_data",
        "recovery_probability": 0.5,
    }


def _calculate_streak_break_risk(
    df: pd.DataFrame, patterns: list[dict]
) -> float:
    """Streak break risk based on recent trends and detected patterns."""
    total_rate = df["completed"].mean()
    recent = df.tail(min(5, len(df)))
    recent_rate = recent["completed"].mean()

    if total_rate > 0:
        drop_ratio = max(0, (total_rate - recent_rate) / total_rate)
    else:
        drop_ratio = 0.5

    pattern_boost = 0.0
    for p in patterns:
        if p["type"] == "declining_streaks":
            pattern_boost += 0.2
        if p["type"] == "burnout_risk":
            pattern_boost += 0.3
        if p["type"] == "negative_momentum":
            pattern_boost += 0.15

    risk = (drop_ratio * 0.6) + pattern_boost
    if recent_rate < 0.3:
        risk = max(risk, 0.6)

    return min(1.0, max(0.0, risk))


def _calculate_motivation_risk(
    df: pd.DataFrame, patterns: list[dict]
) -> float:
    """Motivation risk based on declining trends, rolling slope, burnout."""
    if len(df) >= 5:
        completed_vals = df["completed"].astype(float).values
        rolling = pd.Series(completed_vals).rolling(
            window=min(5, len(completed_vals)), min_periods=1
        ).mean().values

        x = np.arange(len(rolling))
        if len(x) > 1:
            slope = np.polyfit(x, rolling, 1)[0]
        else:
            slope = 0.0
    else:
        slope = 0.0

    slope_risk = max(0, -slope * 10)

    burnout_boost = 0.0
    for p in patterns:
        if p["type"] == "burnout_risk":
            burnout_boost = 0.3
        if p["type"] == "frequent_misses":
            burnout_boost += 0.15

    recent = df.tail(min(5, len(df)))
    miss_ratio = 1 - recent["completed"].mean()

    risk = (slope_risk * 0.4) + (miss_ratio * 0.3) + burnout_boost
    return min(1.0, max(0.0, risk))


def _calculate_inconsistency_index(df: pd.DataFrame) -> float:
    """How variable is the user's completion? 0 = consistent, 1 = random."""
    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    if len(daily) < 3:
        return 0.0

    std = daily.std()
    return min(1.0, float(std) if not np.isnan(std) else 0.0)


def _identify_at_risk_habits(df: pd.DataFrame) -> list[str]:
    """Habits whose recent performance is significantly worse than average."""
    at_risk = []
    for task_name, group in df.groupby("task"):
        if len(group) < 3:
            continue
        overall_rate = group["completed"].mean()
        recent = group.tail(min(5, len(group)))
        recent_rate = recent["completed"].mean()
        if overall_rate > 0.3 and recent_rate < overall_rate * 0.5:
            at_risk.append(str(task_name))
    return at_risk


def _project_streak(df: pd.DataFrame) -> int:
    """
    Project how many more days the current streak will likely last
    based on the exponentially smoothed completion probability.
    """
    if df.empty:
        return 0

    vals = df["completed"].astype(float).values
    if len(vals) < 3:
        return int(vals.sum())

    # Exponential smoothing
    alpha = 0.3
    smoothed = [vals[0]]
    for v in vals[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])

    current_prob = smoothed[-1]
    if current_prob < 0.1:
        return 0

    # Geometric series: expected streak = 1 / (1 - p) capped at 30
    projected = int(min(30, current_prob / (1 - current_prob + 0.01)))
    return max(0, projected)


def _estimate_optimal_focus(df: pd.DataFrame) -> str:
    """
    Estimate the best time window for habit execution.
    Since we don't have timestamp data, we analyze day-of-week patterns
    to recommend the best days to push hard.
    """
    if len(df) < 14:
        return "Insufficient data — need 2+ weeks of tracking."

    df_copy = df.copy()
    df_copy["dow"] = df_copy["date"].dt.day_name()
    dow_rates = df_copy.groupby("dow")["completed"].mean()

    if dow_rates.empty:
        return "N/A"

    best_days = dow_rates.nlargest(3).index.tolist()
    return f"Peak performance days: {', '.join(best_days)}"


def _system_health_score(
    df: pd.DataFrame,
    streak_risk: float,
    motivation_risk: float,
    inconsistency: float,
) -> float:
    """
    Composite health score (0 – 100).
    Factors: overall completion rate, risk scores, consistency.
    """
    completion = df["completed"].mean() * 100
    risk_penalty = (streak_risk + motivation_risk) * 15
    inconsistency_penalty = inconsistency * 20

    score = completion - risk_penalty - inconsistency_penalty
    return max(0.0, min(100.0, score))


def _overall_trend_direction(df: pd.DataFrame) -> str:
    """
    Determine overall trend: 'ascending', 'descending', 'stable',
    or 'volatile'.
    """
    if len(df) < 7:
        return "insufficient_data"

    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    vals = daily.values

    if len(vals) < 5:
        return "insufficient_data"

    x = np.arange(len(vals))
    slope = np.polyfit(x, vals, 1)[0]
    std = np.std(vals)

    if std > 0.4:
        return "volatile"
    elif slope > 0.02:
        return "ascending"
    elif slope < -0.02:
        return "descending"
    return "stable"


def _recovery_probability(df: pd.DataFrame, streak_risk: float) -> float:
    """
    Estimate the probability of recovery based on historical resilience.
    High resilience = quick bouncebacks from bad periods in the past.
    """
    if len(df) < 10:
        return 0.5

    vals = df["completed"].astype(float).values
    # Count recovery events: transition from 0 to 1
    recoveries = sum(
        1 for i in range(1, len(vals))
        if vals[i] == 1 and vals[i - 1] == 0
    )
    # Count failure events: transition from 1 to 0
    failures = sum(
        1 for i in range(1, len(vals))
        if vals[i] == 0 and vals[i - 1] == 1
    )

    if failures == 0:
        return 0.9

    resilience = recoveries / max(failures, 1)
    base_prob = min(1.0, resilience * 0.7)

    # If streak risk is high, lower the recovery probability slightly
    adjusted = base_prob * (1 - streak_risk * 0.3)
    return max(0.0, min(1.0, adjusted))
