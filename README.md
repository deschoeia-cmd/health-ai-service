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

### Model Used

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Source**: Hugging Face
- **Type**: Transformer-based sentence embedding model
- **Properties**:
  - Lightweight
  - Freely available
  - Runs fully locally (no external AI APIs)

The model is downloaded once at container startup and cached locally.

### Input Validation

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

### Classification Approach
The classification logic works as follows:
1. Define a small set of example sentences per category.
2. Load a sentence embedding model at startup.
3. Pre-compute embeddings for all reference example sentences.
4. Embed the incoming user text.
5. Compute cosine similarity between the input embedding and all reference example sentences.
6. Select most similar reference sentence to determine the predicted label.
7. Normalize cosine similarity of the  most similar reference sentence via softmax (temperature=`0.05`) into a probability distribution; the highest probability is returned as the confidence score.
8. Return:
   - the predicted `label` and `confidence`
  > ⚠️ The temperature value has not been fully optimized and may need further tuning for best results.
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

### Docker
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
  "confidence": 0.98
}
```
The confidence value represents the cosine similarity between the input text and the closest category prototype, normalized via softmax (temperature=`0.05`) into a probability distribution.

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
## Using the Interactive API Docs

FastAPI automatically generates an interactive UI for the API. Once the service is running, open your browser and go to:

```text
http://localhost:8000/docs
```

From there:
Click on the endpoint you want to try (`GET /health` or `POST /analyze`).

**For `/health`:**
1. Click on the panel `GET /health`
2. Click **"Try it out"** in the top right of the endpoint panel.
3. Click **"Execute"** to send the request.
4. The response will appear below, showing `"status": "ok"`.

**For `/analyze`:**
1. Click on the panel `POST /analyze`
2. Click **"Try it out"** in the top right of the endpoint panel.
3. Replace the example value in the `text` ("string") field with your own input
4. Click **"Execute"** to send the request.
5. The response will appear below, showing the `label` and `confidence`.

---
## Known Limitation: Negation Handling
### The Problem
The classifier uses cosine similarity over reference sentence embeddings (all-MiniLM-L6-v2). A limitation is that negated symptom descriptions are not always classified correctly.
### For example:

**Request body**
```json
{
  "text": "I don't have high blood pressure."
}
```

**Response**
```json

{
  "label": "urgent_review",
  "confidence": 0.14
}
```

This happens because the embedding model is trained on semantic similarity rather than logical meaning, causing negated and non-negated versions of the same symptom to land close together in vector space.

### Edge Cases to Be Aware Of
Not all negation is equal. There is an important distinction between:

| Sentence | Negation Type | Correct Class |
|---|---|---|
| *"I don't have chest pain"* | Negates the symptom → benign | `low_concern` |
| *"I cannot stay conscious"* | Negates the ability → dangerous | `urgent_review` |
| *"I can't breathe"* | Negates the ability → dangerous | `urgent_review` |
| *"I have no fever"* | Negates the symptom → benign | `low_concern` |

A naive negation rule would incorrectly downgrade "I cannot stay conscious" because it detects the word "not" inside "cannot".

### Possible Solutions

**1. Expand the Example Set**

Add explicit negation examples to the categories, such as:

`low_concern` 

- *"I don't have chest pain"*
- *"I have no shortness of breath"*
- *"I am not experiencing any dizziness"*

`urgent_review`

- *"I can't breath"*
- *"My legs don't work as they used to before"* (to quote Ed Sheeran ;))
- *"I cannot see since two days"*

This pulls the model in the right direction without changing the architecture.


**2. Rule-Based Negation Guard**

Distinguish between **symptom negation** and **ability negation**:
- `no / don't have / do not have` → likely benign, consider downgrading
- `can't / cannot / unable to` → still urgent, do not downgrade

This is recommended as an **immediate safeguard** in a safety-critical context. However there will always be edge cases to consider.


**3. Switch to a Medical-Domain Model**

Replace `all-MiniLM-L6-v2` with a model trained on clinical text, where the distinction
between *"chest pain"* and *"no chest pain"* is explicitly meaningful:
- [`pritamdeka/S-PubMedBert-MS-MARCO`](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO)
- [`microsoft/BiomedNLP-BiomedBERT-base-uncased`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased)

---
