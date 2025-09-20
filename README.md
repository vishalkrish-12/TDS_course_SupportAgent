# TDS Virtual TA (spaCy+FAISS Version)

This project is a FastAPI-based backend for a Virtual Teaching Assistant that answers student questions using a combination of spaCy for text processing and FAISS for efficient vector search. It is designed to be deployed as a containerized web service (e.g., on Render).

---

## Features

- **Question Answering API**: Accepts student questions (optionally with images) and returns answers with relevant links.
- **RAG (Retrieval-Augmented Generation)**: Uses FAISS for fast semantic search over precomputed document embeddings.
- **spaCy NLP**: Utilizes the `en_core_web_md` model for text preprocessing and embedding.
- **Proxy to OpenAI-compatible API**: Uses an API proxy for LLM completions.
- **CORS Enabled**: Ready for use with web frontends.

---

## Project Structure

```
.
├── main.py           # FastAPI app and core logic
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container build instructions
├── .dockerignore     # Files to exclude from Docker image
├── data/             # Precomputed embeddings, metadata, and documents
```

---

## Setup & Deployment

### 1. Requirements

- Python 3.11+
- Docker (for containerized deployment)
- spaCy model: `en_core_web_md`
- Precomputed data files in `data/`:
  - `embeddings.npy`
  - `metadatas.pkl`
  - `documents.pkl`
  - `faiss.index`

### 2. Environment Variable

Set the following environment variable (required for API proxy):

- `AIPIPE_TOKEN`: Your API key for the OpenRouter proxy.

### 3. Build & Run Locally

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

# Run the app
uvicorn main:app --reload
```

### 4. Build & Deploy with Docker

```bash
docker build -t tds-virtual-ta .
docker run -e AIPIPE_TOKEN=your_token_here -p 8000:8000 tds-virtual-ta
```

### 5. Deploy on Render

- Connect your repo to Render.
- Set `AIPIPE_TOKEN` in the environment variables.
- Render will build and run the container automatically.

---

## API Usage

### POST `/api`

**Request:**
```json
{
  "question": "What is FAISS?",
  "image": null
}
```

**Request:**
```bash
curl "http://<your-domain>/api/" `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"I know Docker but have not used Podman before. Should I use Docker for this course?\"}"
```

**Response:**
```json
{
  "answer": "FAISS is a library for efficient similarity search...",
  "links": [
    {"url": "https://github.com/facebookresearch/faiss", "text": "FAISS GitHub"}
  ]
}
```

---

## License

This project is for educational use.

---
