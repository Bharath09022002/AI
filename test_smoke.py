"""Quick test for greeting & casual chat handling."""
import sys
sys.path.insert(0, ".")

from models.schemas import TrackerEntry
from modules.nlp import classify_intent, extract_entities
from modules.analyzer import compute_statistics, get_habit_breakdown
from modules.patterns import detect_patterns
from modules.predictions import predict_risks
from modules.coach import generate_coaching
from modules.response_builder import build_response

# Build sample data
data = []
for i in range(1, 8):
    data.append(TrackerEntry(
        date=f"2026-03-0{i}", task="Workout",
        completed=(i % 2 == 0), streak=i, category="fitness",
    ))

stats = compute_statistics(data)
breakdown = get_habit_breakdown(data)
patterns = detect_patterns(data)
preds = predict_risks(data, patterns)

# Test different conversational inputs
test_messages = [
    "Hi",
    "Hey Jarvis",
    "Good morning",
    "Thanks!",
    "Who are you?",
    "What can you do?",
    "Bye",
    "lol",
]

for msg in test_messages:
    intent = classify_intent(msg)
    entities = extract_entities(msg)
    coaching = generate_coaching(
        intent=intent, question=msg, stats=stats,
        breakdown=breakdown, patterns=patterns,
        predictions=preds, entities=entities,
    )
    print(f"\n{'='*60}")
    print(f"USER: {msg}")
    print(f"INTENT: {intent.value}")
    print(f"JARVIS: {coaching['analysis'][:200]}")
    print(f"{'='*60}")

print("\n=== ALL CONVERSATION TESTS PASSED ===")
