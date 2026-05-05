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
- **Source**: Hugging Face
- **Type**: Transformer-based sentence embedding model
- **Properties**:
  - Lightweight
  - Freely available
  - Runs fully locally (no external AI APIs)

The model is downloaded once at container startup and cached locally.

---

## Classification Approach

The classification logic follows the suggested approach from the challenge:

1. Define a small set of example sentences per category.
2. Load a sentence embedding model at startup.
3. Pre-compute embeddings for all reference example sentences.
4. Embed the incoming user text.
5. Compute cosine similarity between the input embedding and all example embeddings.
6. Select the label of the most similar example sentence.
7. Return:
   - the predicted `label`
   - a similarity-based `confidence` score (cosine similarity)

This keeps the service simple, transparent, and easy to extend.

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
  "confidence": 0.74
}
```
The confidence value represents the cosine similarity between the input text and the closest reference example.

---

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
