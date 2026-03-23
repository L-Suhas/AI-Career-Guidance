from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import os

from recommender import CareerRecommender
from rl_agent import rl_agent

app = FastAPI(title="AI Based Career Guidance System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_store.json")

with open(os.path.join(BASE_DIR, "careers.json"), "r") as f:
    careers_data = json.load(f)

recommender = CareerRecommender(careers_data)

def load_feedback():
    try:
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_feedback_to_file(career_title, action):
    data = load_feedback()
    if career_title not in data:
        data[career_title] = {"accept": 0, "reject": 0}
    data[career_title][action] = data[career_title].get(action, 0) + 1
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

class UserProfile(BaseModel):
    name: str
    degree: str
    major: str
    interests: List[str]
    skills: List[str]
    personality_traits: List[str]
    work_preference: str
    goals: str

class FeedbackModel(BaseModel):
    user_name: str
    career_title: str
    action: str

@app.get("/")
def root():
    return {"message": "AI Based Career Guidance System is running!", "careers_loaded": len(careers_data)}

@app.get("/careers")
def get_all_careers():
    return {"careers": careers_data, "total": len(careers_data)}

@app.post("/recommend")
def recommend_careers(profile: UserProfile):
    recommendations = recommender.get_recommendations(profile.dict(), top_n=5)
    for rec in recommendations:
        rl_bonus = rl_agent.get_bonus(rec["title"])
        rec["match_score"] = round(min(rec["match_score"] + rl_bonus * 10, 99.9), 1)
        rec["rl_boost"] = round(rl_bonus * 100, 1)
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    for i, rec in enumerate(recommendations):
        rec["rank"] = i + 1
    return {
        "student_name": profile.name,
        "recommendations": recommendations,
        "total_careers_analyzed": len(careers_data),
    }

@app.post("/feedback")
def submit_feedback(feedback: FeedbackModel):
    rl_agent.update(feedback.career_title, feedback.action)
    save_feedback_to_file(feedback.career_title, feedback.action)
    return {
        "message": f"Feedback recorded for '{feedback.career_title}'",
        "action": feedback.action,
        "rl_weights": rl_agent.get_all_weights(),
        "status": "success",
    }

@app.get("/career/{career_id}")
def get_career_detail(career_id: int):
    for career in careers_data:
        if career["id"] == career_id:
            return career
    return {"error": "Career not found"}

@app.get("/admin/sessions")
def admin_sessions():
    return {"sessions": [], "message": "Enable PostgreSQL for session tracking"}

@app.get("/admin/feedback")
def admin_feedback():
    feedback_data = load_feedback()
    return {"feedback_stats": feedback_data, "rl_weights": rl_agent.get_all_weights()}

@app.get("/admin/rl-weights")
def rl_weights():
    return {"weights": rl_agent.get_all_weights()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
