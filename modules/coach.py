"""
Coaching module for the AI Coach.
Generates conversational, empathetic coaching responses based on
intent, statistics, patterns, and predictions.
"""

import random
from models.schemas import Intent


def generate_coaching(
    intent: Intent,
    question: str,
    stats: dict,
    breakdown: list[dict],
    patterns: list[dict],
    predictions: dict,
    entities: dict,
) -> dict:
    """
    Generate the coaching output: analysis, insights, suggestions, motivation.

    Returns:
        dict with keys: analysis, insights, suggestions, motivation
    """
    analysis = _build_analysis(intent, question, stats, breakdown, predictions)
    insights = _build_insights(patterns, breakdown, predictions)
    suggestions = _build_suggestions(
        intent, patterns, predictions, breakdown, entities
    )
    motivation = _pick_motivation(intent, stats, predictions, entities)

    return {
        "analysis": analysis,
        "insights": insights,
        "suggestions": suggestions,
        "motivation": motivation,
    }


# ─── Analysis Builder ────────────────────────────────────────────────────────

def _build_analysis(
    intent: Intent,
    question: str,
    stats: dict,
    breakdown: list[dict],
    predictions: dict,
) -> str:
    """Build a human-readable analysis paragraph based on intent."""

    success_rate = stats.get("success_rate", "0%")
    current_streak = stats.get("current_streak", "0 days")
    weekly = stats.get("weekly_performance", "0%")
    risk_summary = predictions.get("risk_summary", "")

    if intent == Intent.WEEKLY_REVIEW:
        return (
            f"This week you're at {weekly} completion rate with an overall "
            f"success rate of {success_rate}. Your current streak stands at "
            f"{current_streak}. {risk_summary} "
            "Let me break down what's working and what needs attention."
        )

    if intent == Intent.STREAK_ANALYSIS:
        longest = stats.get("longest_streak", "0 days")
        streak_risk = predictions.get("streak_break_risk", 0)
        risk_label = (
            "Your streak looks solid!"
            if streak_risk < 0.3
            else "There's some risk of breaking your streak."
            if streak_risk < 0.6
            else "⚠️ Your streak is at risk — let's protect it."
        )
        return (
            f"Your current streak is {current_streak} and your longest ever "
            f"is {longest}. {risk_label} "
            f"Overall success rate: {success_rate}."
        )

    if intent == Intent.FAILURE_ANALYSIS:
        weak = [h for h in breakdown if h["success_rate"] < 50]
        if weak:
            weak_names = ", ".join(h["habit"] for h in weak[:3])
            return (
                f"Your most challenging habits right now are: {weak_names}. "
                f"These have below 50% success rates. Your overall rate is "
                f"{success_rate}. Let's figure out what's going wrong and "
                "how to fix it."
            )
        return (
            f"Actually, you don't have any critically failing habits! "
            f"Your overall success rate is {success_rate}. "
            "That said, there's always room for improvement."
        )

    if intent == Intent.IMPROVEMENT_ADVICE:
        return (
            f"With a {success_rate} overall success rate and {weekly} "
            f"this week, you have good momentum. "
            "I've analyzed your patterns and have some targeted suggestions "
            "to help you level up."
        )

    if intent == Intent.TREND_COMPARISON:
        improving = [h for h in breakdown if h["trend"] == "improving"]
        declining = [h for h in breakdown if h["trend"] == "declining"]
        parts = []
        if improving:
            parts.append(
                f"Improving: {', '.join(h['habit'] for h in improving[:3])}"
            )
        if declining:
            parts.append(
                f"Declining: {', '.join(h['habit'] for h in declining[:3])}"
            )
        trend_text = ". ".join(parts) if parts else "All habits are stable"
        return (
            f"Here's your trend analysis — {trend_text}. "
            f"Overall success rate: {success_rate}."
        )

    if intent == Intent.BURNOUT_CHECK:
        motivation_risk = predictions.get("low_motivation_risk", 0)
        if motivation_risk > 0.6:
            return (
                "I'm seeing signs that you might be pushing too hard. "
                f"Your motivation risk score is elevated at "
                f"{motivation_risk:.0%}. Your body and mind need recovery "
                "to perform at their best."
            )
        return (
            "You seem to be managing your energy well! No strong burnout "
            f"signals detected. Current success rate: {success_rate}."
        )

    if intent == Intent.HABIT_SPECIFIC:
        habit_names = entities.get("habit_names", [])
        relevant = [
            h for h in breakdown
            if any(hn in h["habit"].lower() for hn in habit_names)
        ]
        if relevant:
            h = relevant[0]
            return (
                f"Let's look at {h['habit']}: {h['success_rate']}% success "
                f"rate, streak of {h['current_streak']}, trend is "
                f"{h['trend']}. {_habit_detail(h)}"
            )

    # GENERAL_ADVICE fallback
    return (
        f"Your overall success rate is {success_rate}, you're on a "
        f"{current_streak} streak, and this week's performance is {weekly}. "
        f"{risk_summary}"
    )


def _habit_detail(h: dict) -> str:
    """Generate a mini commentary for a specific habit."""
    if h["trend"] == "improving":
        return "You're on an upward trend — keep the momentum going!"
    if h["trend"] == "declining":
        return (
            "This one is trending downward. Let's think about what "
            "changed and how to get back on track."
        )
    if h["success_rate"] < 40:
        return (
            "This is a tough one for you. Consider breaking it into "
            "smaller chunks or lowering the bar to rebuild consistency."
        )
    return "Solid performance — stay the course!"


# ─── Insights Builder ────────────────────────────────────────────────────────

def _build_insights(
    patterns: list[dict],
    breakdown: list[dict],
    predictions: dict,
) -> list[str]:
    """Build a list of behavioral insight strings."""
    insights: list[str] = []

    for p in patterns:
        if p["type"] == "insufficient_data":
            insights.append(p["description"])
            continue
        if p["type"] == "all_stable":
            insights.append("All your habits are performing consistently.")
            continue

        insights.append(p["description"])

    # Add prediction-based insights
    streak_risk = predictions.get("streak_break_risk", 0)
    if streak_risk > 0.5:
        insights.append(
            f"Streak break risk is {streak_risk:.0%} — "
            "your recent activity shows a dip that could snowball."
        )

    at_risk = predictions.get("at_risk_habits", [])
    if at_risk:
        insights.append(
            f"These habits are at risk of falling off: {', '.join(at_risk)}."
        )

    # Category insights
    categories = {}
    for h in breakdown:
        cat = h.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(h["success_rate"])

    for cat, rates in categories.items():
        avg = sum(rates) / len(rates) if rates else 0
        if avg < 40:
            insights.append(
                f"Your {cat} habits are your weakest category "
                f"(avg {avg:.0f}% success)."
            )
        elif avg > 80:
            insights.append(
                f"Your {cat} habits are your strongest category "
                f"(avg {avg:.0f}% success) — well done!"
            )

    return insights[:8]  # Cap at 8 insights


# ─── Suggestions Builder ─────────────────────────────────────────────────────

_SUGGESTION_TEMPLATES = {
    "reduce_difficulty": [
        "Try reducing the goal for {habit} — "
        "start with a smaller target to rebuild consistency.",
        "Lower the bar for {habit}. Doing 50% of your goal is better "
        "than skipping entirely.",
    ],
    "add_reminders": [
        "Set a daily reminder for {habit} at a time when you're "
        "most likely to follow through.",
        "Pair {habit} with an existing routine — "
        "habit stacking is one of the most effective techniques.",
    ],
    "rest_days": [
        "Consider adding a planned rest day to avoid burnout. "
        "Sustainable consistency beats short bursts.",
        "Your body and mind need recovery. Schedule 1-2 rest days "
        "per week to maintain long-term performance.",
    ],
    "change_timing": [
        "If you're missing {habit} often, try moving it to a "
        "different time of day when you have more energy.",
        "Morning habits tend to have higher completion rates. "
        "Consider shifting {habit} to the morning.",
    ],
    "celebrate_wins": [
        "Don't forget to celebrate your {habit} streak! "
        "Acknowledging progress fuels motivation.",
        "You're doing well on {habit} — "
        "reward yourself for hitting milestones.",
    ],
    "general_boost": [
        "Track your wins at the end of each day. "
        "Reflection boosts motivation and self-awareness.",
        "Find an accountability partner for your weakest habits. "
        "Social commitment increases follow-through by up to 65%.",
        "Visualize your future self completing each habit. "
        "Mental rehearsal primes your brain for action.",
    ],
}


def _build_suggestions(
    intent: Intent,
    patterns: list[dict],
    predictions: dict,
    breakdown: list[dict],
    entities: dict,
) -> list[str]:
    """Build a list of smart, personalized suggestions."""
    suggestions: list[str] = []

    weak_habits = [h for h in breakdown if h["success_rate"] < 50]
    strong_habits = [h for h in breakdown if h["success_rate"] >= 70]

    # Pattern-driven suggestions
    for p in patterns:
        if p["type"] == "burnout_risk":
            suggestions.append(
                random.choice(_SUGGESTION_TEMPLATES["rest_days"])
            )
        if p["type"] == "declining_streaks" and p["affected_habits"]:
            habit = p["affected_habits"][0]
            suggestions.append(
                random.choice(_SUGGESTION_TEMPLATES["reduce_difficulty"])
                .format(habit=habit)
            )
        if p["type"] == "weak_days":
            suggestions.append(p["description"])
        if p["type"] == "frequent_misses" and p["affected_habits"]:
            habit = p["affected_habits"][0]
            suggestions.append(
                random.choice(_SUGGESTION_TEMPLATES["add_reminders"])
                .format(habit=habit)
            )

    # Weak habit suggestions
    for h in weak_habits[:2]:
        suggestions.append(
            random.choice(_SUGGESTION_TEMPLATES["change_timing"])
            .format(habit=h["habit"])
        )

    # Strong habit celebration
    for h in strong_habits[:1]:
        suggestions.append(
            random.choice(_SUGGESTION_TEMPLATES["celebrate_wins"])
            .format(habit=h["habit"])
        )

    # General boosts
    if len(suggestions) < 3:
        suggestions.extend(
            random.sample(
                _SUGGESTION_TEMPLATES["general_boost"],
                min(2, len(_SUGGESTION_TEMPLATES["general_boost"])),
            )
        )

    # Prediction-driven
    if predictions.get("streak_break_risk", 0) > 0.5:
        suggestions.append(
            "Your streak is at risk. Focus on completing just ONE habit "
            "today — that alone can reset your momentum."
        )

    return suggestions[:6]  # Cap at 6 suggestions


# ─── Motivation Picker ────────────────────────────────────────────────────────

_MOTIVATIONAL_MESSAGES = {
    "struggling": [
        "Every expert was once a beginner. The fact that you're tracking "
        "your habits means you're already ahead of 90% of people. "
        "Keep showing up — consistency compounds. 💪",
        "Rough patches are part of the journey, not the end of it. "
        "Tomorrow is a fresh start, and you have the data to "
        "make it count. I believe in you.",
        "You're not failing — you're learning what works and what "
        "doesn't. That's the most valuable data of all. "
        "Let's adjust and come back stronger.",
    ],
    "doing_well": [
        "You're building incredible momentum! Every completed habit "
        "is a vote for the person you're becoming. "
        "Stay locked in — greatness is a habit. 🔥",
        "Consistency is your superpower right now. "
        "The compound effect of your daily actions will surprise "
        "you in a few weeks. Keep going!",
        "You're proof that discipline beats motivation. "
        "Your future self is going to thank you for today's effort.",
    ],
    "neutral": [
        "Progress isn't always linear, and that's okay. "
        "What matters is that you keep showing up. "
        "Small steps, big results. Let's make today count!",
        "You've got the tools, the data, and the drive. "
        "That's everything you need. "
        "Let's turn these insights into action!",
        "Remember: you don't have to be perfect, "
        "just consistent. Every day is a new opportunity "
        "to move one step closer to your goals.",
    ],
}


def _pick_motivation(
    intent: Intent,
    stats: dict,
    predictions: dict,
    entities: dict,
) -> str:
    """Pick a motivational message based on context."""
    sentiment = entities.get("sentiment", "neutral")
    motivation_risk = predictions.get("low_motivation_risk", 0)

    # Determine mood bucket
    if (
        sentiment == "negative"
        or motivation_risk > 0.6
        or intent in (Intent.FAILURE_ANALYSIS, Intent.BURNOUT_CHECK)
    ):
        bucket = "struggling"
    elif (
        sentiment == "positive"
        or motivation_risk < 0.2
        or intent == Intent.IMPROVEMENT_ADVICE
    ):
        bucket = "doing_well"
    else:
        bucket = "neutral"

    return random.choice(_MOTIVATIONAL_MESSAGES[bucket])
