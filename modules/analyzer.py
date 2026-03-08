"""
Data Analysis module for the AI Coach.
Computes comprehensive statistics from tracker data using Pandas and NumPy.
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
    Compute overall and per-habit statistics.

    Returns:
        dict with keys: success_rate, current_streak, longest_streak,
        weekly_performance, monthly_performance, total_habits_tracked,
        habit_difficulty_scores
    """
    df = _to_dataframe(tracker_data)

    if df.empty:
        return {
            "success_rate": "0%",
            "current_streak": "0 days",
            "longest_streak": "0 days",
            "weekly_performance": "0%",
            "monthly_performance": "0%",
            "total_habits_tracked": 0,
            "habit_difficulty_scores": {},
        }

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

    return {
        "success_rate": f"{success_rate:.1f}%",
        "current_streak": f"{current_streak} days",
        "longest_streak": f"{longest_streak} days",
        "weekly_performance": f"{weekly_perf:.1f}%",
        "monthly_performance": f"{monthly_perf:.1f}%",
        "total_habits_tracked": int(len(unique_habits)),
        "habit_difficulty_scores": difficulty_scores,
    }


def get_habit_breakdown(tracker_data: list[TrackerEntry]) -> list[dict]:
    """
    Compute per-habit statistics breakdown.

    Returns:
        List of dicts, each with: habit, category, total_entries,
        completed_count, success_rate, current_streak, trend
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

        breakdown.append({
            "habit": str(task_name),
            "category": str(category),
            "total_entries": int(total),
            "completed_count": int(completed),
            "success_rate": round(rate, 1),
            "current_streak": int(habit_streak),
            "trend": trend,
        })

    # Sort by success rate ascending (worst habits first for coaching)
    breakdown.sort(key=lambda x: x["success_rate"])
    return breakdown


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
