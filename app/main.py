from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI(title="Health AI Service")

# 1) Load a lightweight local model (downloaded once, then cached locally)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # [2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

# 2) Reference examples per category
    # More linguistic variety → improves embedding separation
    # Includes time dimension (“for days”, “not improving”)
    # Captures uncertainty language (“something feels off”)
    # Includes borderline cases → helps classification boundaries
    # More “real patient phrasing”
CATEGORY_EXAMPLES = {
    "low_concern": [
        # clearly benign / everyday
        "I feel fine overall.",
        "Just a mild headache, nothing serious.",
        "I have a slight cold but I'm okay.",
        "I feel a bit tired after work.",
        "I slept poorly but otherwise feel normal.",
        "I have mild muscle soreness from exercise.",
        "I feel a little stressed but physically okay.",
        "I have a runny nose but no other symptoms.",
        "Just minor fatigue, probably from lack of sleep.",
        "I feel okay, just checking in."
    ],

    "needs_follow_up": [
        # persistent, unclear, or worsening
        "I have been feeling tired for over a week.",
        "I feel dizzy on and off for the past few days.",
        "My headache keeps coming back.",
        "I have a cough that is not improving.",
        "I feel unusually weak lately.",
        "I have mild chest discomfort that comes and goes.",
        "I feel nauseous frequently.",
        "My symptoms are not severe but they are persistent.",
        "I feel more short of breath than usual when walking.",
        "Something feels off but I cannot explain it.",
        "I am worried because my symptoms are lasting longer than expected."
    ],

    "urgent_review": [
        # acute, severe, red-flag symptoms
        "I have severe chest pain and shortness of breath.",
        "I feel like I cannot breathe properly.",
        "I suddenly lost strength in my arm.",
        "I am having trouble speaking clearly.",
        "I feel faint and might pass out.",
        "I have intense abdominal pain.",
        "I am coughing up blood.",
        "I have a high fever and feel confused.",
        "I have sudden vision loss.",
        "I feel pressure in my chest spreading to my arm.",
        "I cannot stay conscious.",
        "I am experiencing severe difficulty breathing."
    ]
}

# Pre-compute per-category prototype embeddings at startup (mean of all examples)
category_prototypes: dict[str, torch.Tensor] = {}
for label, texts in CATEGORY_EXAMPLES.items():
    embeddings = model.encode(texts, convert_to_tensor=True)  # (N, D)
    category_prototypes[label] = embeddings.mean(dim=0)       # (D,) prototype vector


class AnalyzeRequest(BaseModel):
    text: str

# Check if input id valid
    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text must not be empty.")
        if len(v.strip()) < 10:
            raise ValueError("Text is too short to analyze.")
        if len(v) > 500:
            raise ValueError("Text is too long (max 500 characters).")
        return v.strip()

@app.get("/health")
def health():
    # Required by the task: returns whether the service is running [1](https://indema-my.sharepoint.com/personal/iara_deschoenmacker_indema_ch/Documents/Microsoft%20Copilot%20Chat-Dateien/Lightweight_AI_REST_Service_Challenge.pdf)
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # Embed user input
    input_embedding = model.encode(req.text, convert_to_tensor=True)

    # Compare against each category prototype using cosine similarity
    scores = {
        label: float(util.cos_sim(input_embedding, proto).item())
        for label, proto in category_prototypes.items()
    }

    best_label = max(scores, key=scores.__getitem__)
    best_conf = scores[best_label]

    return {"label": best_label, "confidence": round(best_conf, 2)}