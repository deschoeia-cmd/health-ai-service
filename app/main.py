from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="Health AI Service")

# 1) Load a lightweight local model (downloaded once, then cached locally)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # [2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

# 2) Reference examples per category (from the challenge PDF)
CATEGORY_EXAMPLES = {
    "low_concern": [
        "I feel well today.",
        "I slept badly but otherwise feel fine.",
        "I have mild tiredness but no other symptoms."
    ],
    "needs_follow_up": [
        "I have been feeling dizzy for several days.",
        "I feel unusually tired and it is not improving.",
        "My symptoms are not severe but I am worried."
    ],
    "urgent_review": [
        "I have chest pain and shortness of breath.",
        "I feel faint and have difficulty breathing.",
        "I have sudden weakness on one side of my body."
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

# Check if input id valid
    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text must not be empty.")
        if len(v.strip()) < 10:
            raise ValueError("Text is too short to analyze.")
        if len(v) > 200:
            raise ValueError("Text is too long (max 1000 characters).")
        return v.strip()

@app.get("/health")
def health():
    # Required by the task: returns whether the service is running [1](https://indema-my.sharepoint.com/personal/iara_deschoenmacker_indema_ch/Documents/Microsoft%20Copilot%20Chat-Dateien/Lightweight_AI_REST_Service_Challenge.pdf)
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # 3) Embed user input
    input_embedding = model.encode(req.text, convert_to_tensor=True)

    # 4) Compare using cosine similarity against all example phrases
    similarities = util.cos_sim(input_embedding, example_embeddings)[0]

    # 5) Find best match
    best_idx = int(similarities.argmax())
    best_label = example_labels[best_idx]
    best_conf = float(similarities[best_idx].item())

    return {"label": best_label, "confidence": round(best_conf, 2)}