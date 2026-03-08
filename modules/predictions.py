"""
Predictive module for the AI Coach.
Predicts streak break risk, low motivation risk, and habit inconsistency.
Uses rolling averages, trend slopes, and recent activity patterns.
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
            - risk_summary: str (human-readable)
            - at_risk_habits: list[str]
    """
    df = _to_dataframe(tracker_data)

    if df.empty or len(df) < 3:
        return {
            "streak_break_risk": 0.0,
            "low_motivation_risk": 0.0,
            "inconsistency_index": 0.0,
            "risk_summary": "Not enough data for predictions yet.",
            "at_risk_habits": [],
        }

    streak_risk = _calculate_streak_break_risk(df, detected_patterns)
    motivation_risk = _calculate_motivation_risk(df, detected_patterns)
    inconsistency = _calculate_inconsistency_index(df)
    at_risk = _identify_at_risk_habits(df)

    # Build summary
    risks = []
    if streak_risk > 0.7:
        risks.append("⚠️ High streak break risk")
    elif streak_risk > 0.4:
        risks.append("⚡ Moderate streak break risk")

    if motivation_risk > 0.7:
        risks.append("⚠️ Low motivation detected")
    elif motivation_risk > 0.4:
        risks.append("⚡ Motivation may be wavering")

    if inconsistency > 0.6:
        risks.append("⚠️ Highly inconsistent behavior")

    summary = "; ".join(risks) if risks else "Looking stable — keep going!"

    return {
        "streak_break_risk": round(streak_risk, 2),
        "low_motivation_risk": round(motivation_risk, 2),
        "inconsistency_index": round(inconsistency, 2),
        "risk_summary": summary,
        "at_risk_habits": at_risk,
    }


def _calculate_streak_break_risk(
    df: pd.DataFrame, patterns: list[dict]
) -> float:
    """
    Streak break risk based on:
    - Recent completion rate (last 3 entries vs overall)
    - Whether a declining pattern was detected
    - Recent streak values trending downward
    """
    total_rate = df["completed"].mean()
    recent = df.tail(min(5, len(df)))
    recent_rate = recent["completed"].mean()

    # Base risk from recent drop
    if total_rate > 0:
        drop_ratio = max(0, (total_rate - recent_rate) / total_rate)
    else:
        drop_ratio = 0.5

    # Pattern boost
    pattern_boost = 0.0
    for p in patterns:
        if p["type"] == "declining_streaks":
            pattern_boost += 0.2
        if p["type"] == "burnout_risk":
            pattern_boost += 0.3

    risk = (drop_ratio * 0.6) + pattern_boost
    # If recent rate is very low, clamp higher
    if recent_rate < 0.3:
        risk = max(risk, 0.6)

    return min(1.0, max(0.0, risk))


def _calculate_motivation_risk(
    df: pd.DataFrame, patterns: list[dict]
) -> float:
    """
    Motivation risk based on:
    - Declining completion trend (slope of rolling average)
    - Burnout pattern presence
    - Ratio of missed entries in the recent window
    """
    # Rolling average slope
    if len(df) >= 5:
        completed_vals = df["completed"].astype(float).values
        rolling = pd.Series(completed_vals).rolling(
            window=min(5, len(completed_vals)), min_periods=1
        ).mean().values

        # Simple linear regression slope on rolling averages
        x = np.arange(len(rolling))
        if len(x) > 1:
            slope = np.polyfit(x, rolling, 1)[0]
        else:
            slope = 0.0
    else:
        slope = 0.0

    # Negative slope = declining motivation
    slope_risk = max(0, -slope * 10)  # Scale up

    # Burnout pattern boost
    burnout_boost = 0.0
    for p in patterns:
        if p["type"] == "burnout_risk":
            burnout_boost = 0.3
        if p["type"] == "frequent_misses":
            burnout_boost += 0.15

    # Recent miss ratio
    recent = df.tail(min(5, len(df)))
    miss_ratio = 1 - recent["completed"].mean()

    risk = (slope_risk * 0.4) + (miss_ratio * 0.3) + burnout_boost
    return min(1.0, max(0.0, risk))


def _calculate_inconsistency_index(df: pd.DataFrame) -> float:
    """
    Inconsistency index: how variable is the user's completion?
    0 = perfectly consistent, 1 = completely random.
    Based on the standard deviation of daily completion rates.
    """
    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    if len(daily) < 3:
        return 0.0

    std = daily.std()
    return min(1.0, float(std) if not np.isnan(std) else 0.0)


def _identify_at_risk_habits(df: pd.DataFrame) -> list[str]:
    """
    Identify habits whose recent performance (last 5 entries)
    is significantly worse than their overall average.
    """
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
