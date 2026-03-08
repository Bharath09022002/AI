"""
Pattern Detection module for the AI Coach — J.A.R.V.I.S. Edition.
Detects behavioral patterns in tracker data — burnout signals,
decline trends, day-of-week weaknesses, habit co-occurrence,
recovery detection, streak velocity shifts, and category imbalances.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from models.schemas import TrackerEntry
from modules.analyzer import _to_dataframe, get_habit_breakdown


def detect_patterns(tracker_data: list[TrackerEntry]) -> list[dict]:
    """
    Analyze tracker data and return a list of detected pattern objects.

    Each pattern has:
        - type: str
        - severity: str ("low", "medium", "high", "critical")
        - description: str (J.A.R.V.I.S.-style human-readable)
        - affected_habits: list[str]
        - confidence: float (0.0 - 1.0)
    """
    df = _to_dataframe(tracker_data)
    if df.empty:
        return [_pattern(
            "insufficient_data", "low",
            "Sir, insufficient data points for pattern analysis. Continue tracking and I will generate insights.",
            [], 0.0
        )]

    patterns: list[dict] = []
    breakdown = get_habit_breakdown(tracker_data)

    # 1. Declining streaks
    declining = [h for h in breakdown if h["trend"] == "declining"]
    if declining:
        names = [h["habit"] for h in declining]
        severity = "critical" if len(declining) >= 3 else "high" if len(declining) >= 2 else "medium"
        confidence = min(1.0, len(declining) * 0.3)
        patterns.append(_pattern(
            "declining_streaks", severity,
            f"Sir, I'm detecting performance degradation in {len(declining)} protocol(s): "
            f"{', '.join(names)}. Recent efficiency metrics are below historical baselines.",
            names, confidence
        ))

    # 2. Frequent missed days — habits below 50% success
    weak = [h for h in breakdown if h["success_rate"] < 50]
    if weak:
        names = [h["habit"] for h in weak]
        patterns.append(_pattern(
            "frequent_misses", "high",
            f"Critical alert, Sir. {len(weak)} protocol(s) operating below 50% efficiency: "
            f"{', '.join(names)}. Immediate parameter adjustment recommended.",
            names, 0.9
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
            f"Positive trajectory confirmed for {', '.join(names)}, Sir. "
            f"Systems showing upward momentum. Maintain current execution parameters.",
            names, 0.8
        ))

    # 6. Strongest vs weakest habits (imbalance)
    if len(breakdown) >= 2:
        strongest = max(breakdown, key=lambda h: h["success_rate"])
        weakest = min(breakdown, key=lambda h: h["success_rate"])
        gap = strongest["success_rate"] - weakest["success_rate"]
        if gap > 30:
            patterns.append(_pattern(
                "habit_imbalance", "medium",
                f"Sir, I'm detecting a {gap:.0f}% efficiency gap between your strongest protocol "
                f"({strongest['habit']} at {strongest['success_rate']}%) and weakest "
                f"({weakest['habit']} at {weakest['success_rate']}%). "
                "Redistributing focus may yield better system-wide results.",
                [strongest["habit"], weakest["habit"]], 0.85
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
            f"Intermittent execution detected for: {', '.join(names)}, Sir. "
            "These protocols are neither failing nor optimal — they fluctuate unpredictably. "
            "Establishing a fixed schedule may stabilize them.",
            names, 0.7
        ))

    # 8. ADVANCED: Momentum shift detection
    momentum_pattern = _detect_momentum_shift(breakdown)
    if momentum_pattern:
        patterns.append(momentum_pattern)

    # 9. ADVANCED: Recovery detection (bounce back after a bad streak)
    recovery_pattern = _detect_recovery(df)
    if recovery_pattern:
        patterns.append(recovery_pattern)

    # 10. ADVANCED: Habit co-occurrence (habits that succeed together)
    cooccurrence = _detect_habit_cooccurrence(df)
    if cooccurrence:
        patterns.append(cooccurrence)

    # 11. ADVANCED: Weekend vs weekday performance
    weekend_pattern = _detect_weekend_effect(df)
    if weekend_pattern:
        patterns.append(weekend_pattern)

    # 12. ADVANCED: Category-level analysis
    category_patterns = _detect_category_imbalance(breakdown)
    patterns.extend(category_patterns)

    if not patterns:
        patterns.append(_pattern(
            "all_stable", "low",
            "All systems are operating within normal parameters, Sir. No anomalies detected.",
            [], 1.0
        ))

    # Sort by confidence descending so the most reliable show first
    patterns.sort(key=lambda p: p.get("confidence", 0), reverse=True)
    return patterns


# ─── Detection Functions ─────────────────────────────────────────────────────

def _detect_burnout(df: pd.DataFrame) -> dict | None:
    """Detect burnout by looking for high activity then sudden drops."""
    if len(df) < 7:
        return None

    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    if len(daily) < 7:
        return None

    daily_vals = daily.values
    recent = daily_vals[-3:].mean() if len(daily_vals) >= 3 else 0
    earlier = daily_vals[-7:-3].mean() if len(daily_vals) >= 7 else 0

    if earlier > 0.7 and recent < 0.4:
        drop = ((earlier - recent) / earlier) * 100
        return _pattern(
            "burnout_risk", "critical",
            f"⚠️ Sir, critical burnout alert. System performance dropped {drop:.0f}% "
            "in the last 72 hours after sustained high-output operations. "
            "I strongly recommend initiating recovery protocols immediately.",
            [], 0.92
        )

    return None


def _detect_day_of_week_weakness(df: pd.DataFrame) -> dict | None:
    """Detect consistently weak days of the week."""
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
        rates = {d: f"{dow_rates[d]*100:.0f}%" for d in weak_days}
        return _pattern(
            "weak_days", "medium",
            f"Sir, telemetry indicates reduced output on: "
            f"{', '.join(f'{d} ({rates[d]})' for d in weak_days)}. "
            "Consider scheduling lower-difficulty tasks on these days or setting automated reminders.",
            [], 0.8
        )

    return None


def _detect_momentum_shift(breakdown: list[dict]) -> dict | None:
    """Detect if overall momentum is strongly positive or negative."""
    if not breakdown:
        return None

    momentums = [h.get("momentum", 0) for h in breakdown]
    avg_momentum = np.mean(momentums)

    if avg_momentum > 25:
        return _pattern(
            "positive_momentum", "low",
            f"Sir, I'm reading a strong positive momentum surge across your protocols "
            f"(avg momentum: +{avg_momentum:.0f}). Your recent output is exceeding historical baselines. "
            "Capitalize on this — conditions are optimal for pushing new challenges.",
            [], 0.85
        )
    elif avg_momentum < -25:
        declining_names = [h["habit"] for h in breakdown if h.get("momentum", 0) < -15]
        return _pattern(
            "negative_momentum", "high",
            f"Warning, Sir. System-wide negative momentum detected "
            f"(avg momentum: {avg_momentum:.0f}). "
            + (f"Most affected: {', '.join(declining_names[:3])}. " if declining_names else "")
            + "Recommend a strategic reset to prevent cascading failures.",
            declining_names[:3], 0.8
        )
    return None


def _detect_recovery(df: pd.DataFrame) -> dict | None:
    """Detect if the user is bouncing back from a bad period."""
    if len(df) < 10:
        return None

    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    if len(daily) < 10:
        return None

    vals = daily.values
    mid_start = max(0, len(vals) - 7)
    mid_end = max(0, len(vals) - 3)
    earlier = vals[mid_start:mid_end].mean() if mid_end > mid_start else 0
    recent = vals[-3:].mean() if len(vals) >= 3 else 0

    if earlier < 0.35 and recent > 0.6:
        return _pattern(
            "recovery_detected", "low",
            "Sir, I'm detecting a system recovery. After a period of reduced output, "
            f"your completion rate has surged from {earlier*100:.0f}% to {recent*100:.0f}% "
            "in the last 3 days. Excellent resilience.",
            [], 0.88
        )
    return None


def _detect_habit_cooccurrence(df: pd.DataFrame) -> dict | None:
    """Detect habits that consistently succeed or fail on the same days."""
    habits = df["task"].unique()
    if len(habits) < 2:
        return None

    pivot = df.pivot_table(
        index=df["date"].dt.date, columns="task",
        values="completed", aggfunc="max", fill_value=0
    )

    if pivot.shape[0] < 7 or pivot.shape[1] < 2:
        return None

    corr = pivot.corr()
    strong_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if not np.isnan(val) and val > 0.6:
                strong_pairs.append(
                    (corr.columns[i], corr.columns[j], val)
                )

    if strong_pairs:
        pair = strong_pairs[0]
        return _pattern(
            "habit_cooccurrence", "low",
            f"Sir, I've found a strong behavioral correlation between '{pair[0]}' and '{pair[1]}' "
            f"(r={pair[2]:.2f}). They tend to succeed or fail together. "
            "You could leverage this by anchoring the weaker one to the stronger.",
            [pair[0], pair[1]], 0.75
        )
    return None


def _detect_weekend_effect(df: pd.DataFrame) -> dict | None:
    """Detect significant performance difference between weekdays and weekends."""
    if len(df) < 14:
        return None

    df_copy = df.copy()
    df_copy["is_weekend"] = df_copy["date"].dt.dayofweek >= 5
    weekday_rate = df_copy[~df_copy["is_weekend"]]["completed"].mean()
    weekend_rate = df_copy[df_copy["is_weekend"]]["completed"].mean()

    diff = abs(weekday_rate - weekend_rate) * 100
    if diff > 25:
        if weekday_rate > weekend_rate:
            return _pattern(
                "weekend_effect", "medium",
                f"Sir, your weekend performance ({weekend_rate*100:.0f}%) drops significantly "
                f"vs weekdays ({weekday_rate*100:.0f}%). A {diff:.0f}% differential. "
                "Consider setting specific weekend-optimized protocols.",
                [], 0.8
            )
        else:
            return _pattern(
                "weekday_effect", "medium",
                f"Interesting, Sir. Your weekday performance ({weekday_rate*100:.0f}%) "
                f"is lower than weekends ({weekend_rate*100:.0f}%). "
                f"A {diff:.0f}% gap. Weekday stress or schedule conflicts may be a factor.",
                [], 0.8
            )
    return None


def _detect_category_imbalance(breakdown: list[dict]) -> list[dict]:
    """Detect if certain habit categories are significantly weaker than others."""
    if not breakdown:
        return []

    categories: dict[str, list[float]] = {}
    for h in breakdown:
        cat = h.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(h["success_rate"])

    if len(categories) < 2:
        return []

    results = []
    avg_rates = {cat: np.mean(rates) for cat, rates in categories.items()}
    best_cat = max(avg_rates, key=avg_rates.get)
    worst_cat = min(avg_rates, key=avg_rates.get)

    if avg_rates[best_cat] - avg_rates[worst_cat] > 25:
        results.append(_pattern(
            "category_imbalance", "medium",
            f"Sir, your '{best_cat}' protocols are at {avg_rates[best_cat]:.0f}% efficiency, "
            f"while '{worst_cat}' protocols are lagging at {avg_rates[worst_cat]:.0f}%. "
            f"A {avg_rates[best_cat]-avg_rates[worst_cat]:.0f}% category differential. "
            "Consider cross-training strategies to level up the weaker category.",
            [], 0.7
        ))

    return results


# ─── Helper ──────────────────────────────────────────────────────────────────

def _pattern(
    ptype: str, severity: str, description: str,
    affected: list[str], confidence: float = 0.5
) -> dict:
    """Helper to create a pattern dict."""
    return {
        "type": ptype,
        "severity": severity,
        "description": description,
        "affected_habits": affected,
        "confidence": round(confidence, 2),
    }
