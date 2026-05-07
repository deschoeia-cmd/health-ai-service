from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F

app = FastAPI(
    title="Health AI Service",
    version="0.1.0",
    description="""
## Description

A lightweight REST API that classifies short health-related text into one of three concern levels.

> ⚠️ Prototype only — not a medical decision tool.
---

**How it works**

Your input is embedded using `all-MiniLM-L6-v2` and matched against reference sentences via cosine similarity.
Cosine similarity scores across all labels are normalized using softmax (temperature=0.05), producing a probability distribution that sums to 1. The label with the highest probability is returned as the result, and its probability is reported as the **confidence score**.
---

**Labels**

| Label | Meaning |
|---|---|
| `low_concern` | Everyday symptoms, no red flags |
| `needs_follow_up` | Persistent or worsening symptoms |
| `urgent_review` | Acute red-flag symptoms |

---

**Getting started**

**GET `/health`:**
Returns whether the service is running.
1. Click on the panel `GET /health`
2. Click **"Try it out"** in the top right of the endpoint panel.
3. Click **"Execute"** to send the request.
4. The response will appear below, showing `"status": "ok"`.

**POST `/analyze`:**
Receives a short health-related text and returns an AI-assisted classification.
1. Click on the panel `POST /analyze`
2. Click **"Try it out"** in the top right of the endpoint panel.
3. Replace the example value in the `text` ("string") field with your own input (10-1000 characters).
4. Click **"Execute"** to send the request.
5. The response will appear below, showing the `label` and `confidence`.

""",
)

# 1) Load a lightweight local model (downloaded once, then cached locally)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # [2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

# 2) Reference examples per category
    # More linguistic variety (including negative sentences e.g., "I don't feel good") → improves embedding separation
    # Includes time dimension (“for days”, “not improving”)
    # Captures uncertainty language (“something feels off”)
    # Includes borderline cases → helps classification boundaries
    # More “real patient phrasing”
CATEGORY_EXAMPLES = {
    "low_concern": [
        # clearly benign / everyday
        "I feel fine overall.",
        "I don't feel chest pain"
        "I don't have any pain for some time now"
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
        "I don't feel good for a month now"
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

# Pre-compute embeddings for category examples once at startup
example_texts = []
example_labels = []
for label, texts in CATEGORY_EXAMPLES.items():
    for t in texts:
        example_texts.append(t)
        example_labels.append(label)

example_embeddings = model.encode(example_texts, convert_to_tensor=True)

class AnalyzeRequest(BaseModel):
    text: str

# Check if input is valid
    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text must not be empty.")
        if len(v.strip()) < 10:
            raise ValueError("Text is too short to analyze.")
        if len(v) > 1000:
            #  input text longer than 256 word pieces is truncated (corresponding to approx. 1000 characters)
            raise ValueError("Text is too long (max 1000 characters).")
        return v.strip()
@app.get("/health")
def health():
    # Return whether the service is running
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    input_embedding = model.encode(req.text, convert_to_tensor=True)
    similarities = util.cos_sim(input_embedding, example_embeddings)[0]

    # Apply softmax to normalize scores into a probability distribution
    temperature = 0.05
    probabilities = F.softmax(similarities / temperature, dim=0)

    best_idx = int(probabilities.argmax())
    best_label = example_labels[best_idx]
    best_conf = float(probabilities[best_idx].item())

    return {"label": best_label, "confidence": round(best_conf, 2)}