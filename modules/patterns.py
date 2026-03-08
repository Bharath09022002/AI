"""
Pattern Detection module for the AI Coach.
Detects behavioral patterns in tracker data — burnout signals,
decline trends, day-of-week weaknesses, strongest/weakest habits, etc.
"""

import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from models.schemas import TrackerEntry
from modules.analyzer import _to_dataframe, get_habit_breakdown


def detect_patterns(tracker_data: list[TrackerEntry]) -> list[dict]:
    """
    Analyze tracker data and return a list of detected pattern objects.

    Each pattern has:
        - type: str (e.g. "declining_streak", "burnout_risk")
        - severity: str ("low", "medium", "high")
        - description: str (human-readable)
        - affected_habits: list[str]
    """
    df = _to_dataframe(tracker_data)
    if df.empty:
        return [_pattern(
            "insufficient_data", "low",
            "Not enough data to detect patterns yet. Keep tracking!",
            []
        )]

    patterns: list[dict] = []
    breakdown = get_habit_breakdown(tracker_data)

    # 1. Declining streaks
    declining = [h for h in breakdown if h["trend"] == "declining"]
    if declining:
        names = [h["habit"] for h in declining]
        severity = "high" if len(declining) >= 3 else "medium"
        patterns.append(_pattern(
            "declining_streaks", severity,
            f"Your performance is declining for: {', '.join(names)}. "
            "Recent completion rates are lower than your earlier averages.",
            names
        ))

    # 2. Frequent missed days — habits below 50% success
    weak = [h for h in breakdown if h["success_rate"] < 50]
    if weak:
        names = [h["habit"] for h in weak]
        patterns.append(_pattern(
            "frequent_misses", "high",
            f"These habits have less than 50% success rate: {', '.join(names)}. "
            "Consider simplifying them or adjusting your goals.",
            names
        ))

    # 3. Burnout detection — high activity followed by sudden drops
    burnout_signals = _detect_burnout(df)
    if burnout_signals:
        patterns.append(burnout_signals)

    # 4. Day-of-week weakness
    dow_pattern = _detect_day_of_week_weakness(df)
    if dow_pattern:
        patterns.append(dow_pattern)

    # 5. Improvement trends
    improving = [h for h in breakdown if h["trend"] == "improving"]
    if improving:
        names = [h["habit"] for h in improving]
        patterns.append(_pattern(
            "improvement_trend", "low",
            f"Great progress on: {', '.join(names)}! "
            "Your recent performance is trending upward.",
            names
        ))

    # 6. Strongest vs weakest habits
    if len(breakdown) >= 2:
        strongest = max(breakdown, key=lambda h: h["success_rate"])
        weakest = min(breakdown, key=lambda h: h["success_rate"])
        if strongest["success_rate"] - weakest["success_rate"] > 30:
            patterns.append(_pattern(
                "habit_imbalance", "medium",
                f"Big gap between your strongest habit "
                f"({strongest['habit']} at {strongest['success_rate']}%) "
                f"and weakest ({weakest['habit']} at {weakest['success_rate']}%). "
                "Consider giving extra attention to the weaker one.",
                [strongest["habit"], weakest["habit"]]
            ))

    # 7. Inconsistency pattern
    inconsistent = [
        h for h in breakdown
        if h["trend"] == "stable" and 40 <= h["success_rate"] <= 70
    ]
    if inconsistent:
        names = [h["habit"] for h in inconsistent]
        patterns.append(_pattern(
            "inconsistent_behavior", "medium",
            f"These habits are inconsistent (neither strong nor weak): "
            f"{', '.join(names)}. You complete them sometimes but not reliably.",
            names
        ))

    if not patterns:
        patterns.append(_pattern(
            "all_stable", "low",
            "No concerning patterns detected. You're on a solid track!",
            []
        ))

    return patterns


def _detect_burnout(df: pd.DataFrame) -> dict | None:
    """
    Detect burnout by looking for a period of high activity
    followed by a sharp drop in completions.
    """
    if len(df) < 7:
        return None

    # Group by date and compute daily completion rate
    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    if len(daily) < 7:
        return None

    daily_vals = daily.values
    # Check the last 3 days vs the 4 days before
    recent = daily_vals[-3:].mean() if len(daily_vals) >= 3 else 0
    earlier = daily_vals[-7:-3].mean() if len(daily_vals) >= 7 else 0

    if earlier > 0.7 and recent < 0.4:
        drop = ((earlier - recent) / earlier) * 100
        return _pattern(
            "burnout_risk", "high",
            f"Possible burnout detected! Your completion dropped {drop:.0f}% "
            "in the last few days after a strong streak. "
            "Consider taking a rest day or reducing habit difficulty.",
            []
        )

    return None


def _detect_day_of_week_weakness(df: pd.DataFrame) -> dict | None:
    """
    Detect if certain days of the week are consistently weaker.
    """
    if len(df) < 14:
        return None

    df_copy = df.copy()
    df_copy["dow"] = df_copy["date"].dt.day_name()
    dow_rates = df_copy.groupby("dow")["completed"].mean()

    day_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    weak_days = [
        day for day in day_names
        if day in dow_rates.index and dow_rates[day] < 0.4
    ]

    if weak_days:
        return _pattern(
            "weak_days", "medium",
            f"You tend to miss habits on {', '.join(weak_days)}. "
            "Try scheduling easier tasks on those days or adding reminders.",
            []
        )

    return None


def _pattern(
    ptype: str, severity: str, description: str, affected: list[str]
) -> dict:
    """Helper to create a pattern dict."""
    return {
        "type": ptype,
        "severity": severity,
        "description": description,
        "affected_habits": affected,
    }
