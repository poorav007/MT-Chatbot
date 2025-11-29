from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    raise SystemExit("Set OPENROUTER_KEY in .env")

# Chroma
DB_DIR = "rag_db"
TOP_K = 3
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("docs")

CASUAL_PHRASES = [
    "hi", "hello", "hey", "yo", "sup", "bye",
    "good morning", "helo", "what's up"
]

# ---------------------------------------------------------------
# REPLACEMENT FOR SentenceTransformer (uses OpenRouter instead)
# ---------------------------------------------------------------
def embed_query(text):
    url = "https://openrouter.ai/api/v1/embeddings"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "baai/bge-large-en-v1.5",  
            "input": text
        }
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def retrieve_context(query, k=TOP_K):
    q_vec = embed_query(query)
    res = collection.query(query_embeddings=[q_vec], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    combined = []
    for d, m in zip(docs, metas):
        src = m.get("source", "unknown")
        chunk_idx = m.get("chunk")
        combined.append(f"Source: {src} (chunk {chunk_idx})\n{d}")

    return "\n\n---\n\n".join(combined)


def openrouter_chat(system_instruction, user_prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "MyRAGApp"
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 5.0,
            "max_tokens": 512
        }
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


class Query(BaseModel):
    question: str


@app.post("/chat")
async def chat_api(data: Query):
    query = data.question.strip()

    if query.lower() in CASUAL_PHRASES:
        return {"answer": "Hello! How can I help you today?"}

    context = retrieve_context(query)

    if not context.strip():
        return {"answer": "Not enough information in the documents."}

    system_instruction = (
        "You must answer ONLY using the provided CONTEXT. "
        "If the answer is not found, reply exactly: 'Not enough information in the documents.' "
        "Do NOT use outside knowledge."
    )

    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    answer = openrouter_chat(system_instruction, prompt)

    return {"answer": answer}

