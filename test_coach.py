"""Quick test script for the AI Coach endpoint."""
import urllib.request
import json

url = "http://localhost:8000/ai/coach"
payload = {
    "question": "How am I doing this week?",
    "tracker_data": [
        {"date": "2026-03-08", "task": "Workout", "completed": True, "streak": 5, "category": "fitness"},
        {"date": "2026-03-07", "task": "Workout", "completed": True, "streak": 4, "category": "fitness"},
        {"date": "2026-03-06", "task": "Reading", "completed": False, "streak": 0, "category": "study"},
        {"date": "2026-03-05", "task": "Meditation", "completed": True, "streak": 10, "category": "health"},
        {"date": "2026-03-04", "task": "Workout", "completed": False, "streak": 0, "category": "fitness"},
        {"date": "2026-03-03", "task": "Reading", "completed": True, "streak": 2, "category": "study"},
        {"date": "2026-03-02", "task": "Meditation", "completed": True, "streak": 9, "category": "health"},
    ]
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
        print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
