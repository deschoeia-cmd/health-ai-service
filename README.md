# Lightweight Health AI REST Service

This repository contains a lightweight AI-powered REST service that classifies short health-related text inputs into one of three concern levels.  
The service runs fully **locally**, uses a **freely available sentence-transformer model**, and is packaged using **Docker**.

> ⚠️ This is a technical prototype for demonstration purposes only and **not a medical decision system**.

---

## Overview

The service exposes a small REST API built with **FastAPI**.  
It accepts a short health check-in text and classifies it into one of the following categories:

- `low_concern`
- `needs_follow_up`
- `urgent_review`

The classification is based on **semantic similarity** between the user input and a small set of reference example sentences.

---

## Model Used

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Source**: Sentence-Transformers
- **Type**: Transformer-based sentence embedding model
- **Properties**:
  - Lightweight
  - Freely available
  - Runs fully locally (no external AI APIs)

The model is downloaded once at container startup and cached locally.

---
## Input Validation

Before any classification is performed, incoming text is validated using a Pydantic field validator. Requests that do not meet the following criteria are rejected with a `422 Unprocessable Entity` response:

| Rule | Requirement |
|------|-------------|
| Not empty | Text must not be blank or whitespace-only |
| Minimum length | Text must be at least 10 characters |
| Maximum length | Text must not exceed 1000 characters (corresponding to approx. 256 tokens) |

**Example rejection response (input too short):**
```json
{
  "detail": [
    {
      "type": "value_error",
      "msg": "Value error, Text is too short to analyze.",
      "input": "ow"
    }
  ]
}
```

## Classification Approach
The classification logic works as follows:
1. Define a small set of example sentences per category.
2. Load a sentence embedding model at startup.
3. Pre-compute embeddings for all reference example sentences.
4. Embed the incoming user text.
5. Compute cosine similarity between the input embedding and all reference example sentences.
6. Select the top-k most similar reference sentences and apply **weighted k-NN voting** to determine the predicted label (default k=5)
7. Return:
   - the predicted `label`

## Calculation of Confidence
The calculation of the "confidence" is as follows:
1. After computing cosine similarities between the input and all reference sentences, the top-5 most similar examples are selected.
2. For each of the top-k examples (default k=5), its cosine similarity score is added to the running total of its corresponding label.
3. The label with the highest accumulated similarity score is the predicted label (weighted k-NN voting).
4. Confidence is calculated as the fraction of the winning label's score over the total accumulated score across all top-k examples.
5. A confidence close to 1.0 means the top-k matches were dominated by a single label. A value close to 0.33 (for 3 labels) indicates an uncertain, evenly distributed result.
6. Return:
   - `confidence` score between 0 and 1.

---

## Project Structure

```text
.
├── app/
│   └── main.py          # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container setup
└── README.md
```
---

## Requirements
Before running the service, make sure you have the following installed:

### Docker (required)
Docker is used to build and run the application in a container.
- Install Docker from the official website: https://www.docker.com/get-started  
- After installation, verify it works:
  ```bash
  docker --version

---

## API Endpoints

### `GET /health`

Returns whether the service is running.

**Response**
```json
{
  "status": "ok"
}
```

### `POST /analyze`

Receives a short health-related text and returns an AI-assisted classification.

**Request body**
```json
{
  "text": "I have felt unusually tired for several days and feel dizzy when I stand up."
}
```

**Response**
```json

{
  "label": "needs_follow_up",
  "confidence": 0.65
}
```
The confidence value represents the cosine similarity between the input text and the closest **category prototype** (weighted k-NN voting).


## Running with Docker
### Build the Docker image
```text
docker build -t health-ai-service .
```
### Run the container
```text
docker run --rm -p 8000:8000 health-ai-service
```
The API will be available at:
```text
http://localhost:8000
```
---

## Testing the Service
### Health check
```text
curl http://localhost:8000/health
```
### Classification example 1
```text

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel fine today. I slept badly, but I have no pain or breathing problems."}'

```
### Classification example 2
```text

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I have chest pain and shortness of breath."}'

```
---
## Using the Interactive API Docs

FastAPI automatically generates an interactive UI for the API. Once the service is running, open your browser and go to:

```text
http://localhost:8000/docs
```

From there:
Click on the endpoint you want to try (`GET /health` or `POST /analyze`).

**For `/analyze`:**
1. Click on the panel `GET /health`
2. Click **"Try it out"** in the top right of the endpoint panel.
3. Click **"Execute"** to send the request.
4. The response will appear below, showing `"status": "ok"`.

**For `/health`:**
1. Click on the panel `POST /analyze`
2. Click **"Try it out"** in the top right of the endpoint panel.
3. Choose the numbre of top-k for the weighted k-NN voting (default is k=5).
4. Replace the example value in the `text` ("string") field with your own input
5. Click **"Execute"** to send the request.
6. The response will appear below, showing the `label` and `confidence`.

---
