"""
Response Builder module for the AI Coach.
Assembles the final structured CoachResponse from all module outputs.
"""

from models.schemas import CoachResponse, StatisticsBlock


def build_response(
    stats: dict,
    coaching: dict,
) -> CoachResponse:
    """
    Assemble the final CoachResponse from computed statistics and
    coaching output.

    Args:
        stats: dict from analyzer.compute_statistics()
        coaching: dict from coach.generate_coaching()

    Returns:
        CoachResponse pydantic model
    """
    statistics = StatisticsBlock(
        success_rate=stats.get("success_rate", "0%"),
        current_streak=stats.get("current_streak", "0 days"),
        longest_streak=stats.get("longest_streak", "0 days"),
        weekly_performance=stats.get("weekly_performance", "0%"),
        monthly_performance=stats.get("monthly_performance", "0%"),
        total_habits_tracked=stats.get("total_habits_tracked", 0),
        habit_difficulty_scores=stats.get("habit_difficulty_scores", {}),
    )

    return CoachResponse(
        analysis=coaching.get("analysis", ""),
        statistics=statistics,
        insights=coaching.get("insights", []),
        suggestions=coaching.get("suggestions", []),
        motivation=coaching.get("motivation", ""),
    )
