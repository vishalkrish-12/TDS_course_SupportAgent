import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
import faiss
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import requests
import re

# import spacy
# spacy.cli.download("en_core_web_md")

# --- Load Data and Models ---


DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadatas.pkl")
DOCS_PATH = os.path.join(DATA_DIR, "documents.pkl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")

# EMBEDDINGS_PATH = "embeddings.npy"
# METADATA_PATH = "metadatas.pkl"
# DOCS_PATH = "documents.pkl"
# FAISS_INDEX_PATH = "faiss.index"

nlp = spacy.load("en_core_web_md")

def preprocess(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove markdown links/images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    # Remove stopwords (optional)
    doc = nlp(text)
    text = ' '.join([token.text for token in doc if not token.is_stop])
    return text

with open(METADATA_PATH, "rb") as f:
    metadatas = pickle.load(f)
with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)
embeddings = np.load(EMBEDDINGS_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)

# --- API and Proxy Configuration ---
API_PROXY_URL = "https://aipipe.org/openrouter/v1/chat/completions"
API_KEY = os.getenv("AIPIPE_TOKEN")
if not API_KEY:
    raise Exception("The AIPIPE_TOKEN environment variable is not set. Please set it before running the app.")

# --- Pydantic Models ---
class StudentRequest(BaseModel):
    question: str
    image: Optional[str] = Field(None, description="Optional base64 encoded image string.")

class Link(BaseModel):
    url: str
    text: str

class ApiResponse(BaseModel):
    answer: str
    links: List[Link]

# --- FastAPI App ---
app = FastAPI(
    title="TDS Virtual TA (spaCy+FAISS Version)",
    description="An API for answering student questions using the aipipe.org proxy.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def query_rag(question: str, image: str = None) -> ApiResponse:
    # # 1. Embed the user's question
    # question_embedding = nlp(question).vector.astype("float32").reshape(1, -1)

    # # Normalize the question embedding
    # question_embedding = question_embedding / np.clip(np.linalg.norm(question_embedding, axis=1, keepdims=True), a_min=1e-10, a_max=None)
    
    # Preprocess the question
    clean_question = preprocess(question)
    # Embed the preprocessed question
    question_embedding = nlp(clean_question).vector.astype("float32").reshape(1, -1)
    # Normalize the question embedding
    question_embedding = question_embedding / np.clip(np.linalg.norm(question_embedding, axis=1, keepdims=True), a_min=1e-10, a_max=None)
    
    # 2. Search FAISS index
    k = 20
    D, I = index.search(question_embedding, k)
    # 3. Retrieve top documents and metadata
    top_docs = []
    top_links = []
    for idx in I[0]:
        doc = documents[idx]
        meta = metadatas[idx]
        top_docs.append(doc)
        if meta.get("source", "").startswith("http"):
            top_links.append(Link(url=meta["source"], text=doc))
    context = "\n\n---\n\n".join(top_docs)

    # 4. Augmentation: Construct the prompt
    prompt_template = f"""You are a helpful teaching assistant. Use the following context to answer the student's question
    CONTEXT:
    {context}

    STUDENT'S QUESTION:
    {question}
    """
    if image:
        prompt_template += "\n\nNOTE: The student attached an image, but image understanding is not supported."

    messages = [
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": prompt_template}
    ]

    # 5. Generation: Call the AIProxy endpoint
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 200
    }

    try:
        response = requests.post(API_PROXY_URL, headers=headers, json=payload)
        response.raise_for_status()
        api_response_json = response.json()
        answer = api_response_json["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling AIProxy: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to get a response from the AI proxy: {e}")
    except (KeyError, IndexError) as e:
        print(f"Error parsing AIProxy response: {e}")
        raise HTTPException(status_code=500, detail="Invalid response format from the AI proxy.")

    return ApiResponse(answer=answer, links=top_links)

@app.post("/api", response_model=ApiResponse)
@app.post("/api/", response_model=ApiResponse, include_in_schema=False)
async def handle_question(request: StudentRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question field cannot be empty.")
    return query_rag(request.question, request.image)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "TDS Virtual TA API is running. Send POST requests to /api/"}