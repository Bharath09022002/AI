"""
Natural Language Processing module for the AI Coach — J.A.R.V.I.S. Edition.
Classifies user intent, extracts entities, and performs sentiment analysis.
Uses keyword matching, regex patterns, synonym expansion, and weighted scoring.
No external LLM required.
"""

import re
from models.schemas import Intent


# ─── Intent Keyword Map with Weights ─────────────────────────────────────────
# Each pattern has a weight (1-3) — more specific patterns get higher weight.

_INTENT_PATTERNS: dict[Intent, list[tuple[str, int]]] = {
    Intent.WEEKLY_REVIEW: [
        (r"\bthis week\b", 3), (r"\bweekly\b", 3), (r"\bhow am i doing\b", 3),
        (r"\bweek\b.*\breview\b", 3), (r"\bweek\b.*\bsummary\b", 3),
        (r"\bprogress\b", 2), (r"\boverall\b", 2), (r"\bhow.*going\b", 2),
        (r"\bupdate\b", 1), (r"\breport\b", 2), (r"\bstatus\b", 2),
        (r"\bhow did i\b", 2), (r"\bdiagnostics\b", 2), (r"\bbrief\b", 2),
    ],
    Intent.STREAK_ANALYSIS: [
        (r"\bstreak\b", 3), (r"\bconsecutive\b", 2), (r"\bin a row\b", 3),
        (r"\bstreak\b.*\bbreak\b", 3), (r"\bstreak\b.*\bbroke\b", 3),
        (r"\blost.*streak\b", 3), (r"\bwhy.*streak\b", 3),
        (r"\bkeep.*going\b", 1), (r"\bchain\b", 2),
    ],
    Intent.FAILURE_ANALYSIS: [
        (r"\bfail\b", 3), (r"\bfailing\b", 3), (r"\bmiss\b", 2),
        (r"\bmissed\b", 2), (r"\bskip\b", 2), (r"\bskipped\b", 2),
        (r"\bworst\b", 3), (r"\bweak\b", 2), (r"\bstruggl\w*\b", 3),
        (r"\bcan'?t\b.*\bdo\b", 3), (r"\bnot doing\b", 2),
        (r"\bfailed\b", 3), (r"\bfalling behind\b", 3),
        (r"\bsuck\b", 2), (r"\bterrible\b", 2), (r"\bawful\b", 2),
        (r"\bproblem\b", 1), (r"\bwrong\b", 2), (r"\bwhy am i\b", 2),
    ],
    Intent.IMPROVEMENT_ADVICE: [
        (r"\bimprov\w*\b", 3), (r"\bbetter\b", 2), (r"\btips?\b", 3),
        (r"\badvice\b", 3), (r"\bhow can i\b", 3), (r"\bhelp me\b", 3),
        (r"\bsuggestion\b", 3), (r"\bwhat should\b", 3), (r"\bboost\b", 2),
        (r"\boptimiz\w*\b", 3), (r"\bproductiv\w*\b", 2),
        (r"\blevel up\b", 3), (r"\bhack\b", 2), (r"\bstrateg\w*\b", 2),
        (r"\bmax\w*\b", 1), (r"\benhance\b", 2), (r"\bupgrade\b", 2),
    ],
    Intent.TREND_COMPARISON: [
        (r"\btrend\b", 3), (r"\bgetting better\b", 3),
        (r"\bgetting worse\b", 3), (r"\bcompare\b", 3),
        (r"\bvs\b", 2), (r"\bversus\b", 2),
        (r"\bimproving\b", 2), (r"\bdeclining\b", 3),
        (r"\bover time\b", 3), (r"\bchanged\b", 2),
        (r"\blast week\b", 2), (r"\bbefore\b", 1),
        (r"\bhistory\b", 2), (r"\bpast\b", 1),
    ],
    Intent.HABIT_SPECIFIC: [
        (r"\bworkout\b", 3), (r"\bexercise\b", 3), (r"\breading\b", 3),
        (r"\bmeditat\w*\b", 3), (r"\bstudy\w*\b", 3), (r"\bcoding\b", 3),
        (r"\bwater\b", 2), (r"\bsleep\b", 3), (r"\bdiet\b", 3),
        (r"\bjournal\w*\b", 3), (r"\bwalk\w*\b", 2), (r"\brun\w*\b", 2),
        (r"\byoga\b", 3), (r"\bstretch\w*\b", 3), (r"\bcook\w*\b", 2),
        (r"\bclean\w*\b", 2), (r"\bgym\b", 3), (r"\bpush.?ups?\b", 3),
        (r"\bpull.?ups?\b", 3), (r"\bcardio\b", 3), (r"\bfitness\b", 3),
        (r"\bprayer\b", 3), (r"\bwrit\w*\b", 2), (r"\bpiano\b", 3),
        (r"\bguitar\b", 3), (r"\bpractice\b", 2), (r"\bsavu\b", 3),
    ],
    Intent.BURNOUT_CHECK: [
        (r"\bburnout\b", 3), (r"\btired\b", 3), (r"\bexhaust\w*\b", 3),
        (r"\boverwhelm\w*\b", 3), (r"\btoo much\b", 3),
        (r"\bcan'?t keep up\b", 3), (r"\bstress\w*\b", 3),
        (r"\bmotivat\w*\b", 2), (r"\bdrain\w*\b", 2),
        (r"\bfatig\w*\b", 3), (r"\bbreak\b", 1), (r"\brest\b", 2),
        (r"\benergy\b", 2), (r"\bgive up\b", 3),
    ],
    Intent.GREETING: [
        (r"^hi\b", 5), (r"^hey\b", 5), (r"^hello\b", 5),
        (r"^yo\b", 5), (r"^sup\b", 5), (r"^hola\b", 5),
        (r"^greetings\b", 5), (r"^howdy\b", 5),
        (r"\bgood morning\b", 5), (r"\bgood afternoon\b", 5),
        (r"\bgood evening\b", 5), (r"\bgood night\b", 4),
        (r"^what'?s up\b", 5), (r"^whats up\b", 5),
        (r"^wassup\b", 5), (r"^hii+\b", 5), (r"^heyy+\b", 5),
        (r"\bhow are you\b", 4), (r"\bwhat'?s good\b", 4),
        (r"^jarvis\b", 4), (r"^j\.?a\.?r\.?v\.?i\.?s\b", 4),
    ],
    Intent.CASUAL_CHAT: [
        (r"\bthank\w*\b", 4), (r"\bthanks\b", 4),
        (r"\bbye\b", 4), (r"\bgoodbye\b", 4), (r"\bsee you\b", 4),
        (r"\blater\b", 2), (r"\btake care\b", 4),
        (r"\bwho are you\b", 5), (r"\bwhat are you\b", 5),
        (r"\bwhat can you do\b", 5), (r"\bhelp\b", 3),
        (r"\bwhat do you know\b", 5), (r"\btell me about yourself\b", 5),
        (r"\byou'?re? (?:cool|awesome|great|amazing)\b", 4),
        (r"\blol\b", 3), (r"\bhaha\b", 3), (r"\bnice\b", 2),
        (r"\bokay\b", 2), (r"\bok\b", 2), (r"\bcool\b", 2),
    ],
}


def classify_intent(question: str) -> Intent:
    """
    Classify the user's question into an intent category.
    Uses weighted scoring for more accurate classification.
    Falls back to GENERAL_ADVICE if no strong match.
    """
    question_lower = question.lower().strip()
    scores: dict[Intent, float] = {}

    for intent, patterns in _INTENT_PATTERNS.items():
        score = sum(
            weight for pattern, weight in patterns
            if re.search(pattern, question_lower)
        )
        if score > 0:
            scores[intent] = score

    if not scores:
        return Intent.GENERAL_ADVICE

    # If the top score is very close to the second, we may have multi-intent
    # For now, return the highest scoring intent
    return max(scores, key=scores.get)  # type: ignore


def extract_entities(question: str) -> dict:
    """
    Extract useful entities from the user's question.
    Returns a dict with optional keys: habit_names, time_range, sentiment,
    intensity, comparison_type.
    """
    question_lower = question.lower().strip()
    entities: dict = {}

    # ── Extract habit names ──────────────────────────────────────────────
    habit_keywords = [
        "workout", "exercise", "reading", "meditation", "study",
        "coding", "water", "sleep", "diet", "journaling", "walking",
        "running", "yoga", "stretching", "cooking", "cleaning",
        "gym", "push ups", "pull ups", "pushups", "pullups",
        "cardio", "fitness", "prayer", "writing", "piano",
        "guitar", "practice", "savu",
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
    elif re.search(r"\blast week\b", question_lower):
        entities["time_range"] = "last_week"
    elif re.search(r"\blast month\b", question_lower):
        entities["time_range"] = "last_month"

    # ── Detect sentiment (enhanced) ──────────────────────────────────────
    negative_words = {
        "fail": 2, "bad": 1, "terrible": 2, "awful": 2, "worst": 2,
        "hate": 2, "struggle": 2, "frustrated": 2, "can't": 1, "never": 1,
        "suck": 2, "horrible": 2, "depressed": 3, "hopeless": 3,
        "give up": 3, "quit": 2, "impossible": 2, "disaster": 2,
    }
    positive_words = {
        "great": 2, "good": 1, "awesome": 2, "amazing": 2, "proud": 2,
        "love": 1, "crushing": 2, "nailed": 2, "best": 2,
        "excellent": 2, "fantastic": 2, "perfect": 2, "happy": 1,
        "thriving": 2, "winning": 2, "killing it": 3,
    }

    neg_score = sum(
        weight for word, weight in negative_words.items()
        if word in question_lower
    )
    pos_score = sum(
        weight for word, weight in positive_words.items()
        if word in question_lower
    )

    if neg_score > pos_score:
        entities["sentiment"] = "negative"
    elif pos_score > neg_score:
        entities["sentiment"] = "positive"
    else:
        entities["sentiment"] = "neutral"

    # ── Detect intensity ─────────────────────────────────────────────────
    intensity_markers = [
        "very", "really", "extremely", "super", "so",
        "incredibly", "absolutely", "seriously", "totally",
    ]
    intensity_count = sum(1 for m in intensity_markers if m in question_lower)
    if intensity_count >= 2:
        entities["intensity"] = "high"
    elif intensity_count == 1:
        entities["intensity"] = "moderate"
    else:
        entities["intensity"] = "normal"

    return entities
