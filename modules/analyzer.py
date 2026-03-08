"""
Data Analysis module for the AI Coach — J.A.R.V.I.S. Edition.
Computes comprehensive statistics from tracker data using Pandas and NumPy.
Includes: rolling averages, momentum scoring, habit correlations,
day-of-week analysis, and advanced difficulty metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.schemas import TrackerEntry


def _to_dataframe(tracker_data: list[TrackerEntry]) -> pd.DataFrame:
    """Convert tracker entries to a Pandas DataFrame with proper types."""
    records = [entry.model_dump() for entry in tracker_data]
    if not records:
        return pd.DataFrame(
            columns=["date", "task", "completed", "streak", "category"]
        )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date", ascending=True)
    return df


def compute_statistics(tracker_data: list[TrackerEntry]) -> dict:
    """
    Compute overall and per-habit statistics with advanced metrics.

    Returns:
        dict with keys: success_rate, current_streak, longest_streak,
        weekly_performance, monthly_performance, total_habits_tracked,
        habit_difficulty_scores, momentum_score, best_day, worst_day,
        rolling_7d_avg, consistency_grade, habit_correlations
    """
    df = _to_dataframe(tracker_data)

    if df.empty:
        return _empty_stats()

    total = len(df)
    completed = df["completed"].sum()
    success_rate = (completed / total * 100) if total > 0 else 0

    # ── Streak calculation (overall) ─────────────────────────────────────
    current_streak = _calculate_current_streak(df)
    longest_streak = _calculate_longest_streak(df)

    # ── Weekly performance ───────────────────────────────────────────────
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_df = df[df["date"] >= pd.Timestamp(week_start.date())]
    weekly_total = len(week_df)
    weekly_completed = week_df["completed"].sum() if weekly_total > 0 else 0
    weekly_perf = (
        (weekly_completed / weekly_total * 100) if weekly_total > 0 else 0
    )

    # ── Monthly performance ──────────────────────────────────────────────
    month_start = today.replace(day=1)
    month_df = df[df["date"] >= pd.Timestamp(month_start.date())]
    monthly_total = len(month_df)
    monthly_completed = month_df["completed"].sum() if monthly_total > 0 else 0
    monthly_perf = (
        (monthly_completed / monthly_total * 100) if monthly_total > 0 else 0
    )

    # ── Per-habit difficulty scores ──────────────────────────────────────
    unique_habits = df["task"].unique()
    difficulty_scores = {}
    for habit in unique_habits:
        difficulty_scores[habit] = calculate_difficulty_score(
            df[df["task"] == habit]
        )

    # ── ADVANCED: Momentum Score (-100 to +100) ──────────────────────────
    momentum = _calculate_momentum(df)

    # ── ADVANCED: Day-of-Week Analysis ───────────────────────────────────
    best_day, worst_day, dow_breakdown = _day_of_week_analysis(df)

    # ── ADVANCED: Rolling 7-day average ──────────────────────────────────
    rolling_avg = _rolling_average(df, window=7)

    # ── ADVANCED: Consistency Grade (A+ to F) ────────────────────────────
    consistency_grade = _consistency_grade(df)

    # ── ADVANCED: Habit Correlation Matrix ────────────────────────────────
    correlations = _habit_correlations(df)

    return {
        "success_rate": f"{success_rate:.1f}%",
        "current_streak": f"{current_streak} days",
        "longest_streak": f"{longest_streak} days",
        "weekly_performance": f"{weekly_perf:.1f}%",
        "monthly_performance": f"{monthly_perf:.1f}%",
        "total_habits_tracked": int(len(unique_habits)),
        "habit_difficulty_scores": difficulty_scores,
        # ── Advanced metrics ─────────────────────────
        "momentum_score": momentum,
        "best_day": best_day,
        "worst_day": worst_day,
        "dow_breakdown": dow_breakdown,
        "rolling_7d_avg": rolling_avg,
        "consistency_grade": consistency_grade,
        "habit_correlations": correlations,
    }


def _empty_stats() -> dict:
    return {
        "success_rate": "0%",
        "current_streak": "0 days",
        "longest_streak": "0 days",
        "weekly_performance": "0%",
        "monthly_performance": "0%",
        "total_habits_tracked": 0,
        "habit_difficulty_scores": {},
        "momentum_score": 0,
        "best_day": "N/A",
        "worst_day": "N/A",
        "dow_breakdown": {},
        "rolling_7d_avg": "0%",
        "consistency_grade": "N/A",
        "habit_correlations": {},
    }


def get_habit_breakdown(tracker_data: list[TrackerEntry]) -> list[dict]:
    """
    Compute per-habit statistics breakdown with advanced metrics.

    Returns:
        List of dicts, each with: habit, category, total_entries,
        completed_count, success_rate, current_streak, trend,
        momentum, best_window_rate, worst_window_rate, streak_velocity
    """
    df = _to_dataframe(tracker_data)

    if df.empty:
        return []

    breakdown = []
    for task_name, group in df.groupby("task"):
        total = len(group)
        completed = group["completed"].sum()
        rate = (completed / total * 100) if total > 0 else 0

        # Determine trend: compare first half vs second half completion
        mid = len(group) // 2
        if mid > 0:
            first_half_rate = group.iloc[:mid]["completed"].mean()
            second_half_rate = group.iloc[mid:]["completed"].mean()
            if second_half_rate > first_half_rate + 0.1:
                trend = "improving"
            elif second_half_rate < first_half_rate - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Current streak for this habit
        habit_streak = _calculate_current_streak(group)

        category = (
            group["category"].mode().iloc[0]
            if not group["category"].mode().empty
            else "other"
        )

        # ── ADVANCED: Per-habit momentum ─────────────────────────────────
        habit_momentum = _calculate_momentum(group)

        # ── ADVANCED: Best and worst sliding window ──────────────────────
        best_w, worst_w = _best_worst_window(group, window=7)

        # ── ADVANCED: Streak velocity ────────────────────────────────────
        streak_vel = _streak_velocity(group)

        breakdown.append({
            "habit": str(task_name),
            "category": str(category),
            "total_entries": int(total),
            "completed_count": int(completed),
            "success_rate": round(rate, 1),
            "current_streak": int(habit_streak),
            "trend": trend,
            "momentum": habit_momentum,
            "best_window_rate": best_w,
            "worst_window_rate": worst_w,
            "streak_velocity": streak_vel,
        })

    # Sort by success rate ascending (worst habits first for coaching)
    breakdown.sort(key=lambda x: x["success_rate"])
    return breakdown


# ─── Core Streak Helpers ─────────────────────────────────────────────────────

def calculate_difficulty_score(habit_df: pd.DataFrame) -> float:
    """
    Calculate a difficulty score (1.0 – 10.0) for a habit.
    Higher = harder for the user. Based on failure rate, inconsistency,
    and streak fragility.
    """
    if habit_df.empty:
        return 5.0

    total = len(habit_df)
    completed = habit_df["completed"].sum()
    failure_rate = 1 - (completed / total) if total > 0 else 0.5

    # Streak fragility: how often the streak resets
    streaks = habit_df["streak"].tolist()
    resets = sum(
        1 for i in range(1, len(streaks))
        if streaks[i] < streaks[i - 1]
    )
    reset_ratio = resets / max(len(streaks) - 1, 1)

    # Inconsistency: variance in daily completion
    consistency = habit_df["completed"].astype(int).rolling(
        window=min(7, len(habit_df)), min_periods=1
    ).mean().std()
    consistency = float(consistency) if not np.isnan(consistency) else 0.3

    # Weighted score
    score = (failure_rate * 5) + (reset_ratio * 3) + (consistency * 2)
    return round(max(1.0, min(10.0, score)), 1)


def _calculate_current_streak(df: pd.DataFrame) -> int:
    """Calculate the current active streak from most recent entries."""
    if df.empty:
        return 0

    sorted_df = df.sort_values("date", ascending=False)
    streak = 0
    for _, row in sorted_df.iterrows():
        if row["completed"]:
            streak += 1
        else:
            break
    return streak


def _calculate_longest_streak(df: pd.DataFrame) -> int:
    """Calculate the longest streak in the dataset."""
    if df.empty:
        return 0

    sorted_df = df.sort_values("date", ascending=True)
    longest = 0
    current = 0
    for _, row in sorted_df.iterrows():
        if row["completed"]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# ─── Advanced Metric Helpers ─────────────────────────────────────────────────

def _calculate_momentum(df: pd.DataFrame) -> int:
    """
    Momentum Score (-100 to +100).
    Compares the most recent 25% of entries against the overall average.
    Positive = accelerating, Negative = decelerating.
    """
    if len(df) < 4:
        return 0

    overall_rate = df["completed"].mean()
    window = max(3, len(df) // 4)
    recent_rate = df.tail(window)["completed"].mean()

    if overall_rate == 0:
        return 0

    diff = (recent_rate - overall_rate) / overall_rate
    return int(max(-100, min(100, diff * 100)))


def _day_of_week_analysis(df: pd.DataFrame) -> tuple[str, str, dict]:
    """
    Analyze completion rates by day of the week.
    Returns (best_day_name, worst_day_name, {day: rate%}).
    """
    if len(df) < 7:
        return "N/A", "N/A", {}

    df_copy = df.copy()
    df_copy["dow"] = df_copy["date"].dt.day_name()
    dow_rates = df_copy.groupby("dow")["completed"].mean()

    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    present_days = [d for d in day_order if d in dow_rates.index]
    if not present_days:
        return "N/A", "N/A", {}

    breakdown = {d: round(float(dow_rates[d]) * 100, 1) for d in present_days}
    best = max(present_days, key=lambda d: dow_rates[d])
    worst = min(present_days, key=lambda d: dow_rates[d])
    return best, worst, breakdown


def _rolling_average(df: pd.DataFrame, window: int = 7) -> str:
    """Calculate the rolling N-day average completion rate."""
    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    if len(daily) < window:
        avg = daily.mean() if len(daily) > 0 else 0
    else:
        avg = daily.iloc[-window:].mean()
    return f"{float(avg) * 100:.1f}%"


def _consistency_grade(df: pd.DataFrame) -> str:
    """
    Grade the user's consistency from A+ to F, based on the standard
    deviation of their daily completion rates.
    Lower std = more consistent = higher grade.
    """
    daily = df.groupby(df["date"].dt.date)["completed"].mean()
    if len(daily) < 3:
        return "N/A"

    std = float(daily.std())
    mean_rate = float(daily.mean())

    if np.isnan(std):
        return "N/A"

    # Combined score: high mean + low std = best
    score = mean_rate - (std * 0.5)

    if score >= 0.9:
        return "A+"
    elif score >= 0.8:
        return "A"
    elif score >= 0.7:
        return "B+"
    elif score >= 0.6:
        return "B"
    elif score >= 0.5:
        return "C+"
    elif score >= 0.4:
        return "C"
    elif score >= 0.3:
        return "D"
    else:
        return "F"


def _habit_correlations(df: pd.DataFrame) -> dict:
    """
    Compute pairwise completion correlations between habits.
    Returns dict of { "Habit A ↔ Habit B": correlation_value }.
    High positive correlation = these habits succeed/fail together.
    """
    if len(df["task"].unique()) < 2:
        return {}

    # Pivot: rows=dates, columns=habits, values=completed (0/1)
    pivot = df.pivot_table(
        index=df["date"].dt.date,
        columns="task",
        values="completed",
        aggfunc="max",
        fill_value=0,
    )

    if pivot.shape[1] < 2 or pivot.shape[0] < 5:
        return {}

    corr_matrix = pivot.corr()
    results = {}
    habits = list(corr_matrix.columns)

    for i in range(len(habits)):
        for j in range(i + 1, len(habits)):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val) and abs(val) > 0.3:
                key = f"{habits[i]} ↔ {habits[j]}"
                results[key] = round(float(val), 2)

    return results


def _best_worst_window(
    df: pd.DataFrame, window: int = 7
) -> tuple[float, float]:
    """
    Find the best and worst sliding window completion rates.
    Returns (best_rate%, worst_rate%).
    """
    if len(df) < window:
        rate = df["completed"].mean() * 100 if len(df) > 0 else 0
        return round(rate, 1), round(rate, 1)

    vals = df["completed"].astype(float).values
    rolling = pd.Series(vals).rolling(window=window, min_periods=window).mean()
    rolling = rolling.dropna()

    if rolling.empty:
        return 0.0, 0.0

    return round(float(rolling.max()) * 100, 1), round(float(rolling.min()) * 100, 1)


def _streak_velocity(df: pd.DataFrame) -> str:
    """
    Streak velocity: how fast is the user building streaks?
    Compares recent streak lengths to earlier ones.
    Returns: "accelerating", "decelerating", or "steady".
    """
    streaks = df["streak"].tolist()
    if len(streaks) < 6:
        return "steady"

    mid = len(streaks) // 2
    early_avg = np.mean(streaks[:mid]) if mid > 0 else 0
    late_avg = np.mean(streaks[mid:]) if mid > 0 else 0

    if late_avg > early_avg * 1.2:
        return "accelerating"
    elif late_avg < early_avg * 0.8:
        return "decelerating"
    return "steady"
