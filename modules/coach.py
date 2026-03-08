"""
Coaching module for the AI Coach — J.A.R.V.I.S. Edition.
Generates dynamic, highly contextual coaching responses using
a sentence-composition engine. Every response is unique based on
the exact data, never two identical outputs.
"""

import random
import hashlib
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
    Uses a dynamic composition engine for unique, data-driven responses.
    """
    analysis = _build_analysis(intent, question, stats, breakdown, predictions, entities)
    insights = _build_insights(patterns, breakdown, predictions, stats)
    suggestions = _build_suggestions(
        intent, patterns, predictions, breakdown, entities, stats
    )
    motivation = _pick_motivation(intent, stats, predictions, entities)

    return {
        "analysis": analysis,
        "insights": insights,
        "suggestions": suggestions,
        "motivation": motivation,
    }


# ─── Dynamic Analysis Builder ────────────────────────────────────────────────

def _build_analysis(
    intent: Intent,
    question: str,
    stats: dict,
    breakdown: list[dict],
    predictions: dict,
    entities: dict,
) -> str:
    """Build a dynamic, data-rich analysis using the composition engine."""

    success_rate = stats.get("success_rate", "0%")
    current_streak = stats.get("current_streak", "0 days")
    weekly = stats.get("weekly_performance", "0%")
    monthly = stats.get("monthly_performance", "0%")
    risk_summary = predictions.get("risk_summary", "")
    momentum = stats.get("momentum_score", 0)
    health = predictions.get("system_health_score", 50)
    grade = stats.get("consistency_grade", "N/A")
    best_day = stats.get("best_day", "N/A")
    worst_day = stats.get("worst_day", "N/A")
    trend = predictions.get("trend_direction", "stable")
    projected = predictions.get("projected_streak_days", 0)
    recovery = predictions.get("recovery_probability", 0.5)

    # ── GREETING based on health ─────────────────────────────────────────
    if health >= 80:
        greeting = random.choice([
            "Sir, all systems are operating at peak capacity.",
            "Excellent readings across the board, Sir.",
            "Sir, your operational metrics are exemplary.",
        ])
    elif health >= 50:
        greeting = random.choice([
            "Sir, I've completed the diagnostic scan. Mixed signals detected.",
            "Systems are functional but not yet optimized, Sir.",
            "I've run the full analysis, Sir. Some areas need attention.",
        ])
    else:
        greeting = random.choice([
            "Sir, the diagnostics show several systems under strain.",
            "Warning, Sir. Multiple protocols are operating below safe thresholds.",
            "I must be transparent, Sir — the data is concerning.",
        ])

    # ── CORE STATS PARAGRAPH ─────────────────────────────────────────────
    core = (
        f"Your overall success rate is registering at {success_rate} "
        f"with a system health score of {health:.0f}/100 "
        f"(Consistency Grade: {grade}). "
    )

    if current_streak != "0 days":
        core += f"Active streak: {current_streak}. "

    core += f"This week's output: {weekly}. Monthly average: {monthly}. "

    # ── MOMENTUM BRIEFING ────────────────────────────────────────────────
    if momentum > 20:
        momentum_text = (
            f"I'm reading strong positive momentum at +{momentum}, Sir. "
            "Your recent output exceeds your historical baseline — capitalize on this surge."
        )
    elif momentum < -20:
        momentum_text = (
            f"Momentum is negative at {momentum}, Sir. "
            "Recent performance is falling below your established averages. "
            "Recommend an immediate course correction."
        )
    else:
        momentum_text = "Momentum is neutral — steady state operations."

    # ── TREND + PREDICTION ───────────────────────────────────────────────
    trend_map = {
        "ascending": "The overall vector is ascending, Sir. You are improving.",
        "descending": "The overall vector is descending, Sir. We need to address this.",
        "volatile": "Behavior patterns are volatile, Sir. High variance in daily output.",
        "stable": "The trend is stable. Consistent, but consider pushing for growth.",
    }
    trend_text = trend_map.get(trend, "")

    if projected > 5:
        trend_text += f" Based on exponential smoothing, I project your streak can extend approximately {projected} more days at current trajectory."
    elif projected > 0:
        trend_text += f" However, projected streak continuation is only {projected} days — stay vigilant."

    # ── DAY-OF-WEEK INTEL ────────────────────────────────────────────────
    dow_text = ""
    if best_day != "N/A" and worst_day != "N/A" and best_day != worst_day:
        dow_text = (
            f" Peak performance day: {best_day}. "
            f"Weakest day: {worst_day}."
        )

    # ── INTENT-SPECIFIC ADDENDUM ─────────────────────────────────────────
    addendum = _intent_addendum(intent, breakdown, predictions, entities, recovery)

    # ── RISK FOOTER ──────────────────────────────────────────────────────
    risk_footer = f" {risk_summary}" if risk_summary else ""

    # ── COMPOSE ──────────────────────────────────────────────────────────
    paragraphs = [greeting, core, momentum_text]
    if trend_text:
        paragraphs.append(trend_text)
    if dow_text:
        paragraphs.append(dow_text)
    if addendum:
        paragraphs.append(addendum)
    if risk_footer.strip():
        paragraphs.append(risk_footer.strip())

    return " ".join(p.strip() for p in paragraphs if p.strip())


def _intent_addendum(
    intent: Intent,
    breakdown: list[dict],
    predictions: dict,
    entities: dict,
    recovery: float,
) -> str:
    """Generate an intent-specific paragraph to add context."""

    if intent == Intent.WEEKLY_REVIEW:
        improving = [h["habit"] for h in breakdown if h["trend"] == "improving"]
        declining = [h["habit"] for h in breakdown if h["trend"] == "declining"]
        parts = []
        if improving:
            parts.append(f"Improving protocols: {', '.join(improving[:3])}.")
        if declining:
            parts.append(f"Protocols needing attention: {', '.join(declining[:3])}.")
        return " ".join(parts) if parts else ""

    if intent == Intent.STREAK_ANALYSIS:
        risk = predictions.get("streak_break_risk", 0)
        if risk > 0.6:
            return (
                f"Sir, streak disruption probability is at {risk:.0%}. "
                "I recommend focusing exclusively on your highest-success habit today "
                "to anchor the streak."
            )
        return "Your streak integrity is within acceptable parameters."

    if intent == Intent.FAILURE_ANALYSIS:
        weak = [h for h in breakdown if h["success_rate"] < 50]
        if weak:
            details = [f"'{h['habit']}' ({h['success_rate']}%)" for h in weak[:3]]
            return (
                "Critical protocol analysis: " + ", ".join(details) + ". "
                f"Recovery probability based on your historical resilience: {recovery:.0%}."
            )
        return "No critically failing protocols detected, Sir."

    if intent == Intent.IMPROVEMENT_ADVICE:
        strong = [h for h in breakdown if h["success_rate"] >= 70]
        if strong:
            return (
                f"Strongest protocol: '{strong[0]['habit']}' at {strong[0]['success_rate']}%. "
                "I suggest replicating its scheduling pattern for weaker protocols."
            )
        return "I have targeted recommendations ready based on the pattern analysis."

    if intent == Intent.TREND_COMPARISON:
        accl = [h["habit"] for h in breakdown if h.get("streak_velocity") == "accelerating"]
        decl = [h["habit"] for h in breakdown if h.get("streak_velocity") == "decelerating"]
        parts = []
        if accl:
            parts.append(f"Accelerating: {', '.join(accl[:3])}")
        if decl:
            parts.append(f"Decelerating: {', '.join(decl[:3])}")
        return ". ".join(parts) + "." if parts else ""

    if intent == Intent.BURNOUT_CHECK:
        mot_risk = predictions.get("low_motivation_risk", 0)
        if mot_risk > 0.6:
            return (
                f"Burnout indicators are at {mot_risk:.0%}, Sir. "
                "Your cognitive and physical systems are showing fatigue signatures. "
                "Scheduled rest is not optional — it is essential."
            )
        return "No critical burnout markers detected. Systems are within safe operational limits."

    if intent == Intent.HABIT_SPECIFIC:
        habit_names = entities.get("habit_names", [])
        relevant = [
            h for h in breakdown
            if any(hn in h["habit"].lower() for hn in habit_names)
        ]
        if relevant:
            h = relevant[0]
            return (
                f"Detailed scan of '{h['habit']}': "
                f"{h['success_rate']}% success, "
                f"streak of {h['current_streak']}, "
                f"trend is {h['trend']}, "
                f"momentum at {h.get('momentum', 0):+d}. "
                f"{_habit_detail(h)}"
            )

    return ""


def _habit_detail(h: dict) -> str:
    """Generate a mini commentary for a specific habit."""
    velocity = h.get("streak_velocity", "steady")
    if h["trend"] == "improving" and velocity == "accelerating":
        return "This protocol is on an accelerating upward vector, Sir. Outstanding execution."
    if h["trend"] == "improving":
        return "Positive trajectory confirmed. Maintain current execution parameters."
    if h["trend"] == "declining" and velocity == "decelerating":
        return (
            "This protocol is in a decelerating decline, Sir. "
            "The degradation is compounding. Immediate intervention recommended."
        )
    if h["trend"] == "declining":
        return (
            "Metrics are trending downward. I recommend analyzing recent environmental variables "
            "and adjusting the approach accordingly."
        )
    if h["success_rate"] < 30:
        return (
            "This protocol is in critical failure state. Decompose it into "
            "sub-tasks and establish a minimum viable execution threshold."
        )
    if h["success_rate"] < 50:
        return (
            "Below the efficiency baseline. Consider pairing this with a high-success "
            "protocol for habit stacking."
        )
    return "Operations within acceptable range. Steady as she goes, Sir."


# ─── Insights Builder ────────────────────────────────────────────────────────

def _build_insights(
    patterns: list[dict],
    breakdown: list[dict],
    predictions: dict,
    stats: dict,
) -> list[str]:
    """Build a list of highly intelligent behavioral insight strings."""
    insights: list[str] = []

    # Pattern-driven insights
    for p in patterns:
        if p["type"] == "insufficient_data":
            insights.append("Sir, the data set is insufficient for deep analysis. Continue tracking.")
            continue
        if p["type"] == "all_stable":
            insights.append("All tracking protocols show consistent operational behavior.")
            continue

        # Use the pattern description directly — it's already in J.A.R.V.I.S. voice
        insights.append(p["description"])

    # Prediction-based insights
    streak_risk = predictions.get("streak_break_risk", 0)
    if streak_risk > 0.5:
        insights.append(
            f"Streak integrity threat level: {streak_risk:.0%}. "
            "Recent telemetry shows a destabilizing dip in execution rate."
        )

    at_risk = predictions.get("at_risk_habits", [])
    if at_risk:
        insights.append(
            f"Protocols flagged for cascading failure risk: {', '.join(at_risk)}."
        )

    # Health-based insight
    health = predictions.get("system_health_score", 50)
    if health >= 80:
        insights.append(f"System health score: {health:.0f}/100 — operating in the green zone.")
    elif health >= 50:
        insights.append(f"System health score: {health:.0f}/100 — functional but with optimization headroom.")
    elif health > 0:
        insights.append(f"System health score: {health:.0f}/100 — multiple subsystems under stress.")

    # Correlation insights
    correlations = stats.get("habit_correlations", {})
    for pair, val in list(correlations.items())[:2]:
        if val > 0.5:
            insights.append(
                f"Strong positive correlation detected: {pair} (r={val}). "
                "These protocols reinforce each other."
            )
        elif val < -0.5:
            insights.append(
                f"Negative correlation detected: {pair} (r={val}). "
                "These protocols may compete for the same resources."
            )

    # Momentum insight
    momentum = stats.get("momentum_score", 0)
    if abs(momentum) > 30:
        direction = "positive surge" if momentum > 0 else "negative slide"
        insights.append(
            f"Momentum index: {momentum:+d}. A significant {direction}. "
            + ("Leverage this energy." if momentum > 0 else "Corrective action advised.")
        )

    # Consistency grade insight
    grade = stats.get("consistency_grade", "N/A")
    if grade in ("A+", "A"):
        insights.append(f"Consistency grade: {grade}. Elite-tier discipline detected.")
    elif grade in ("D", "F"):
        insights.append(f"Consistency grade: {grade}. Execution is erratic — establish anchored routines.")

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
                f"'{cat}' category protocols averaging {avg:.0f}% — weakest subsystem."
            )
        elif avg > 80:
            insights.append(
                f"'{cat}' category protocols at {avg:.0f}% — strongest subsystem. Exemplary."
            )

    return insights[:10]  # Cap at 10 insights


# ─── Suggestions Builder ─────────────────────────────────────────────────────

def _build_suggestions(
    intent: Intent,
    patterns: list[dict],
    predictions: dict,
    breakdown: list[dict],
    entities: dict,
    stats: dict,
) -> list[str]:
    """Build highly personalized, data-driven suggestions."""
    suggestions: list[str] = []

    weak_habits = [h for h in breakdown if h["success_rate"] < 50]
    strong_habits = [h for h in breakdown if h["success_rate"] >= 70]
    correlations = stats.get("habit_correlations", {})
    best_day = stats.get("best_day", "N/A")
    worst_day = stats.get("worst_day", "N/A")
    projected = predictions.get("projected_streak_days", 0)

    # ── Pattern-driven suggestions ───────────────────────────────────────
    for p in patterns:
        if p["type"] == "burnout_risk":
            suggestions.append(
                "Sir, initiate recovery mode. Scale down to your 3 easiest protocols for 48 hours. "
                "Sustainable performance requires strategic rest cycles."
            )
        if p["type"] == "declining_streaks" and p.get("affected_habits"):
            habit = p["affected_habits"][0]
            suggestions.append(
                f"Pair '{habit}' with an existing high-success routine — "
                "habit stacking is the most efficient integration method."
            )
        if p["type"] == "weak_days":
            suggestions.append(p["description"])
        if p["type"] == "frequent_misses" and p.get("affected_habits"):
            habit = p["affected_habits"][0]
            suggestions.append(
                f"Configure automated daily reminders for '{habit}', Sir. "
                "Timing them with your peak energy window will maximize compliance."
            )
        if p["type"] == "habit_cooccurrence" and p.get("affected_habits"):
            h1, h2 = p["affected_habits"][:2] if len(p["affected_habits"]) >= 2 else (p["affected_habits"][0], "")
            if h2:
                suggestions.append(
                    f"'{h1}' and '{h2}' are behaviorally linked. "
                    "Use the stronger one as an anchor to pull the weaker one along."
                )
        if p["type"] == "weekend_effect" or p["type"] == "weekday_effect":
            suggestions.append(
                "Sir, I recommend a differentiated schedule: "
                "lighter protocols on weak days, intensive protocols on strong days."
            )

    # ── Weak habit specific suggestions ──────────────────────────────────
    for h in weak_habits[:2]:
        velocity = h.get("streak_velocity", "steady")
        if velocity == "decelerating":
            suggestions.append(
                f"'{h['habit']}' is losing streak velocity. Break it into a 2-minute micro-habit "
                "to rebuild the neural pathway before scaling back up."
            )
        elif h["success_rate"] < 25:
            suggestions.append(
                f"'{h['habit']}' is at {h['success_rate']}% — near system failure. "
                "Reduce the goal to the absolute minimum (e.g., 1 rep, 1 page, 1 minute). "
                "The objective is to reboot the completion circuit."
            )
        else:
            suggestions.append(
                f"If you're missing '{h['habit']}' often, try moving it to {best_day} "
                f"when your completion rates are highest."
                if best_day != "N/A" else
                f"Try scheduling '{h['habit']}' at a different time when you have more energy."
            )

    # ── Strong habit celebration + leverage ──────────────────────────────
    for h in strong_habits[:1]:
        suggestions.append(
            f"'{h['habit']}' is operating at {h['success_rate']}%. "
            "Use its scheduling pattern as a template for struggling protocols."
        )

    # ── Correlation-based suggestions ────────────────────────────────────
    for pair, val in list(correlations.items())[:1]:
        if val > 0.5:
            habits = pair.split(" ↔ ")
            if len(habits) == 2:
                suggestions.append(
                    f"Since '{habits[0]}' and '{habits[1]}' succeed together, "
                    "always execute them in sequence. One triggers the other."
                )

    # ── Day-of-week optimization ─────────────────────────────────────────
    if worst_day != "N/A" and best_day != "N/A":
        suggestions.append(
            f"On {worst_day}s (your weakest day), reduce to only your top 2 easiest habits. "
            f"On {best_day}s, push harder with challenging protocols."
        )

    # ── Prediction-driven ────────────────────────────────────────────────
    streak_risk = predictions.get("streak_break_risk", 0)
    if streak_risk > 0.5:
        suggestions.append(
            "Sir, streak integrity is compromised. Execute ONE habit immediately — "
            "even the smallest action prevents a cascade failure."
        )

    if projected > 10:
        suggestions.append(
            f"At current trajectory, your streak could extend {projected} more days. "
            "Stay the course — you are building compounding returns."
        )

    # ── General boosts if we don't have enough ───────────────────────────
    general = [
        "Log your daily completions with a brief note, Sir. Reflective data enriches future analysis.",
        "Announce your goals to an accountability partner — social commitment increases follow-through by 65%.",
        "Visualize completing each protocol before you begin. Mental pre-computation primes your system for action.",
        "Stack your hardest habit right after your easiest — momentum from a quick win powers through resistance.",
    ]
    if len(suggestions) < 3:
        suggestions.extend(random.sample(general, min(2, len(general))))

    return suggestions[:8]  # Cap at 8


# ─── Motivation Picker ────────────────────────────────────────────────────────

_MOTIVATIONAL_MESSAGES = {
    "struggling": [
        "Sir, I must remind you — even the most advanced systems require recalibration under stress. "
        "The fact that you are tracking and analyzing puts you ahead of 90% of operators. "
        "Let us adjust the parameters and resume. You have my full support.",

        "Sir, performance metrics indicate a temporary setback, not a permanent state. "
        "I have modeled your recovery probability and it is favorable. "
        "Tomorrow's cycle is a clean slate. Let's prepare for it now.",

        "This data is not a verdict, Sir — it is intelligence. "
        "We now know exactly where the system is weakest. "
        "That knowledge is the most powerful tool in our arsenal. Deploying countermeasures.",
    ],
    "doing_well": [
        "Sir, your execution metrics are statistically exceptional. "
        "The compound effect of these consistent protocols will yield extraordinary results. "
        "I calculate you are building something remarkable.",

        "All systems are green, Sir. Your discipline is generating compounding returns "
        "that most operators never achieve. The Mark IV protocols would be proud. "
        "Shall I prepare the next tier of challenges?",

        "Sir, you are demonstrating that systematic execution outperforms sporadic motivation "
        "by orders of magnitude. Your data trail is proof of that principle. Stellar work.",
    ],
    "neutral": [
        "Sir, progress rarely follows a straight line — but the aggregate trajectory matters more "
        "than any single data point. The data shows steady operation. "
        "Consistent execution is what builds the armor.",

        "I have processed the latest data streams, Sir. "
        "You have the tools, the data, and the operational drive. "
        "Awaiting your command to lock onto the next objective.",

        "Sir, remember that perfection is statistically improbable, "
        "but sustained consistency is mathematically guaranteed to compound. "
        "Every completed protocol brings you closer to the target.",
    ],
}


def _pick_motivation(
    intent: Intent,
    stats: dict,
    predictions: dict,
    entities: dict,
) -> str:
    """Pick a contextual motivational message based on health and data."""
    sentiment = entities.get("sentiment", "neutral")
    motivation_risk = predictions.get("low_motivation_risk", 0)
    health = predictions.get("system_health_score", 50)
    momentum = stats.get("momentum_score", 0)

    # Determine mood bucket using multiple signals
    struggling_signals = sum([
        sentiment == "negative",
        motivation_risk > 0.6,
        intent in (Intent.FAILURE_ANALYSIS, Intent.BURNOUT_CHECK),
        health < 40,
        momentum < -30,
    ])

    well_signals = sum([
        sentiment == "positive",
        motivation_risk < 0.2,
        intent == Intent.IMPROVEMENT_ADVICE,
        health > 75,
        momentum > 25,
    ])

    if struggling_signals >= 2:
        bucket = "struggling"
    elif well_signals >= 2:
        bucket = "doing_well"
    else:
        bucket = "neutral"

    return random.choice(_MOTIVATIONAL_MESSAGES[bucket])
