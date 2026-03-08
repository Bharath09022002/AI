"""
Natural Language Processing module for the AI Coach.
Classifies user intent and extracts entities from natural language questions.
No external LLM required — uses keyword matching and regex patterns.
"""

import re
from models.schemas import Intent


# ─── Intent Keyword Map ──────────────────────────────────────────────────────

_INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.WEEKLY_REVIEW: [
        r"\bthis week\b", r"\bweekly\b", r"\bhow am i doing\b",
        r"\bweek\b.*\breview\b", r"\bweek\b.*\bsummary\b",
        r"\bprogress\b", r"\boverall\b", r"\bhow.*going\b",
    ],
    Intent.STREAK_ANALYSIS: [
        r"\bstreak\b", r"\bconsecutive\b", r"\bin a row\b",
        r"\bstreak\b.*\bbreak\b", r"\bstreak\b.*\bbroke\b",
        r"\blost.*streak\b", r"\bwhy.*streak\b",
    ],
    Intent.FAILURE_ANALYSIS: [
        r"\bfail\b", r"\bfailing\b", r"\bmiss\b", r"\bmissed\b",
        r"\bskip\b", r"\bskipped\b", r"\bworst\b", r"\bweak\b",
        r"\bstruggl\b", r"\bcan'?t\b.*\bdo\b", r"\bnot doing\b",
        r"\bfailed\b", r"\bfalling behind\b",
    ],
    Intent.IMPROVEMENT_ADVICE: [
        r"\bimprov\b", r"\bbetter\b", r"\btips?\b", r"\badvice\b",
        r"\bhow can i\b", r"\bhelp me\b", r"\bsuggestion\b",
        r"\bwhat should\b", r"\bboost\b", r"\boptimize\b",
        r"\bproductiv\b",
    ],
    Intent.TREND_COMPARISON: [
        r"\btrend\b", r"\bgetting better\b", r"\bgetting worse\b",
        r"\bcompare\b", r"\bvs\b", r"\bversus\b",
        r"\bimproving\b", r"\bdeclining\b", r"\bover time\b",
    ],
    Intent.HABIT_SPECIFIC: [
        r"\bworkout\b", r"\bexercise\b", r"\breading\b",
        r"\bmeditat\b", r"\bstudy\b", r"\bcoding\b",
        r"\bwater\b", r"\bsleep\b", r"\bdiet\b",
    ],
    Intent.BURNOUT_CHECK: [
        r"\bburnout\b", r"\btired\b", r"\bexhaust\b",
        r"\boverwhelm\b", r"\btoo much\b", r"\bcan'?t keep up\b",
        r"\bstress\b", r"\bmotivat\b",
    ],
}


def classify_intent(question: str) -> Intent:
    """
    Classify the user's question into an intent category.
    Returns the intent with the highest keyword match count.
    Falls back to GENERAL_ADVICE if no strong match.
    """
    question_lower = question.lower().strip()
    scores: dict[Intent, int] = {}

    for intent, patterns in _INTENT_PATTERNS.items():
        score = sum(
            1 for pattern in patterns
            if re.search(pattern, question_lower)
        )
        if score > 0:
            scores[intent] = score

    if not scores:
        return Intent.GENERAL_ADVICE

    return max(scores, key=scores.get)  # type: ignore


def extract_entities(question: str) -> dict:
    """
    Extract useful entities from the user's question.
    Returns a dict with optional keys: habit_name, time_range, sentiment.
    """
    question_lower = question.lower().strip()
    entities: dict = {}

    # ── Extract habit names ──────────────────────────────────────────────
    habit_keywords = [
        "workout", "exercise", "reading", "meditation", "study",
        "coding", "water", "sleep", "diet", "journaling", "walking",
        "running", "yoga", "stretching", "cooking", "cleaning",
    ]
    found_habits = [h for h in habit_keywords if h in question_lower]
    if found_habits:
        entities["habit_names"] = found_habits

    # ── Extract time range ───────────────────────────────────────────────
    if re.search(r"\bthis week\b", question_lower):
        entities["time_range"] = "week"
    elif re.search(r"\bthis month\b", question_lower):
        entities["time_range"] = "month"
    elif re.search(r"\btoday\b", question_lower):
        entities["time_range"] = "today"
    elif re.search(r"\byesterday\b", question_lower):
        entities["time_range"] = "yesterday"
    elif match := re.search(r"\blast (\d+) days?\b", question_lower):
        entities["time_range"] = f"last_{match.group(1)}_days"

    # ── Detect sentiment ─────────────────────────────────────────────────
    negative_words = [
        "fail", "bad", "terrible", "awful", "worst", "hate",
        "struggle", "frustrated", "can't", "never",
    ]
    positive_words = [
        "great", "good", "awesome", "amazing", "proud",
        "love", "crushing", "nailed", "best",
    ]

    neg_count = sum(1 for w in negative_words if w in question_lower)
    pos_count = sum(1 for w in positive_words if w in question_lower)

    if neg_count > pos_count:
        entities["sentiment"] = "negative"
    elif pos_count > neg_count:
        entities["sentiment"] = "positive"
    else:
        entities["sentiment"] = "neutral"

    return entities
