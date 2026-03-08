"""
Pydantic models for the AI Coach API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class HabitCategory(str, Enum):
    FITNESS = "fitness"
    STUDY = "study"
    HEALTH = "health"
    PRODUCTIVITY = "productivity"
    MINDFULNESS = "mindfulness"
    SOCIAL = "social"
    FINANCE = "finance"
    CREATIVITY = "creativity"
    OTHER = "other"


class Intent(str, Enum):
    WEEKLY_REVIEW = "weekly_review"
    STREAK_ANALYSIS = "streak_analysis"
    FAILURE_ANALYSIS = "failure_analysis"
    IMPROVEMENT_ADVICE = "improvement_advice"
    TREND_COMPARISON = "trend_comparison"
    HABIT_SPECIFIC = "habit_specific"
    BURNOUT_CHECK = "burnout_check"
    GENERAL_ADVICE = "general_advice"


# ─── Request Models ──────────────────────────────────────────────────────────

class TrackerEntry(BaseModel):
    """A single tracker data point for one habit on one day."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    task: str = Field(..., description="Habit or task name")
    completed: bool = Field(..., description="Whether the habit was completed")
    streak: Optional[int] = Field(0, description="Current streak count")
    category: Optional[HabitCategory] = Field(
        HabitCategory.OTHER, description="Habit category"
    )


class CoachRequest(BaseModel):
    """The request body for POST /ai/coach."""
    question: str = Field(
        ..., description="Natural language question from the user"
    )
    tracker_data: list[TrackerEntry] = Field(
        ..., description="List of tracker entries to analyze"
    )


# ─── Response Models ─────────────────────────────────────────────────────────

class StatisticsBlock(BaseModel):
    """Computed statistics about the user's tracker data."""
    success_rate: str = Field(..., description="Overall success rate as percentage")
    current_streak: str = Field(..., description="Current active streak")
    longest_streak: str = Field(..., description="Longest streak achieved")
    weekly_performance: str = Field(..., description="This week's completion %")
    monthly_performance: str = Field(..., description="This month's completion %")
    total_habits_tracked: int = Field(..., description="Number of unique habits")
    habit_difficulty_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Difficulty score (1-10) per habit"
    )


class CoachResponse(BaseModel):
    """The structured response from the AI Coach."""
    analysis: str = Field(..., description="Human-readable analysis summary")
    statistics: StatisticsBlock = Field(
        ..., description="Computed statistics"
    )
    insights: list[str] = Field(
        ..., description="List of behavioral insights"
    )
    suggestions: list[str] = Field(
        ..., description="List of smart suggestions"
    )
    motivation: str = Field(
        ..., description="Motivational coaching message"
    )
