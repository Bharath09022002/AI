"""Quick smoke test for the upgraded AI engine."""
import sys
sys.path.insert(0, ".")

from models.schemas import TrackerEntry, Intent
from modules.nlp import classify_intent, extract_entities
from modules.analyzer import compute_statistics, get_habit_breakdown
from modules.patterns import detect_patterns
from modules.predictions import predict_risks
from modules.coach import generate_coaching
from modules.response_builder import build_response

# Build sample data: 2 habits over 7 days
data = []
for i in range(1, 8):
    data.append(TrackerEntry(
        date=f"2026-03-0{i}", task="Workout",
        completed=(i % 2 == 0), streak=i, category="fitness",
    ))
    data.append(TrackerEntry(
        date=f"2026-03-0{i}", task="Reading",
        completed=(i % 3 != 0), streak=i, category="study",
    ))

# Run the full pipeline
stats = compute_statistics(data)
breakdown = get_habit_breakdown(data)
patterns = detect_patterns(data)
preds = predict_risks(data, patterns)

question = "How am I doing this week?"
intent = classify_intent(question)
entities = extract_entities(question)

coaching = generate_coaching(
    intent=intent, question=question, stats=stats,
    breakdown=breakdown, patterns=patterns,
    predictions=preds, entities=entities,
)
resp = build_response(stats=stats, coaching=coaching)

print("=== AI ENGINE SMOKE TEST ===")
print(f"Intent: {intent}")
print(f"Health: {preds.get('system_health_score')}")
print(f"Grade: {stats.get('consistency_grade')}")
print(f"Momentum: {stats.get('momentum_score')}")
print(f"Trend: {preds.get('trend_direction')}")
print(f"Projected Streak: {preds.get('projected_streak_days')} days")
print(f"Recovery Prob: {preds.get('recovery_probability')}")
print(f"Correlations: {stats.get('habit_correlations')}")
print()
print(f"Analysis ({len(resp.analysis)} chars):")
print(resp.analysis[:300])
print()
print(f"Insights ({len(resp.insights)}):")
for i, ins in enumerate(resp.insights):
    print(f"  {i+1}. {ins}")
print()
print(f"Suggestions ({len(resp.suggestions)}):")
for i, s in enumerate(resp.suggestions):
    print(f"  {i+1}. {s}")
print()
print(f"Motivation:")
print(resp.motivation[:200])
print()
print("=== PASSED ===")
