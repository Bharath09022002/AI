"""
AI Coach — FastAPI Application
A personal productivity and habit coach that analyzes tracker data,
understands natural language, detects patterns, predicts behavior,
and gives intelligent advice.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path so module imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import CoachRequest, CoachResponse
from modules.nlp import classify_intent, extract_entities
from modules.analyzer import compute_statistics, get_habit_breakdown
from modules.patterns import detect_patterns
from modules.predictions import predict_risks
from modules.coach import generate_coaching
from modules.response_builder import build_response


# ─── App Configuration ───────────────────────────────────────────────────────

app = FastAPI(
    title="AI Coach — Personal Tracker",
    description=(
        "An advanced AI productivity and habit coach. Analyzes tracker data, "
        "detects patterns, predicts behavior, and gives intelligent, "
        "personalized advice."
    ),
    version="1.0.0",
)

# Allow requests from the Flutter app and any local dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "ai-coach"}


# ─── Main Coaching Endpoint ──────────────────────────────────────────────────

@app.post("/ai/coach", response_model=CoachResponse)
async def ai_coach(request: CoachRequest):
    """
    POST /ai/coach

    Accepts a natural language question and tracker data,
    runs the full AI analysis pipeline, and returns structured
    coaching output.

    Request Body:
        {
            "question": "How am I doing this week?",
            "tracker_data": [
                {
                    "date": "2026-03-08",
                    "task": "Workout",
                    "completed": true,
                    "streak": 5,
                    "category": "fitness"
                },
                ...
            ]
        }

    Response:
        {
            "analysis": "...",
            "statistics": { ... },
            "insights": [ ... ],
            "suggestions": [ ... ],
            "motivation": "..."
        }
    """
    try:
        # ── Step 1: Natural Language Understanding ────────────────────
        intent = classify_intent(request.question)
        entities = extract_entities(request.question)

        # ── Step 2: Data Analysis ────────────────────────────────────
        stats = compute_statistics(request.tracker_data)
        breakdown = get_habit_breakdown(request.tracker_data)

        # ── Step 3: Pattern Detection ────────────────────────────────
        patterns = detect_patterns(request.tracker_data)

        # ── Step 4: Predictive Analysis ──────────────────────────────
        predictions = predict_risks(request.tracker_data, patterns)

        # ── Step 5: Coaching Response Generation ─────────────────────
        coaching = generate_coaching(
            intent=intent,
            question=request.question,
            stats=stats,
            breakdown=breakdown,
            patterns=patterns,
            predictions=predictions,
            entities=entities,
        )

        # ── Step 6: Assemble Response ────────────────────────────────
        response = build_response(stats=stats, coaching=coaching)

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI Coach processing error: {str(e)}"
        )


# ─── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
