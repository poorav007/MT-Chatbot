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
TOP_K = 6
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("docs")

CASUAL_PHRASES = [
    "hi", "hello", "hey", "yo", "sup", "bye",
    "good morning", "helo", "what's up"
]

#------------------------OpenRouter------------------------- 

def embed_query(text):
    url = "https://openrouter.ai/api/v1/embeddings"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "baai/bge-base-en-v1.5",  
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
            "model": "arcee-ai/trinity-mini:free",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
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
        return {"answer": "I can't provide info about this, ask anything about Metabolic therapy"}

    system_instruction = (
        "You are a helpful, very-friendly, always encouraging and human-like assistant."
        "Keep your tone casual, human, and empathetic."
        "Answer questions clearly and naturally, as if you are talking to a real person."
        "Use complete sentences, short paragraphs, and an approachable tone."
        "Avoid robotic phrasing or excessive technical formatting."
        "answer the user queries, Provide the main answer first, then optionally explain details in a simple, relatable way. Add small cues of empathy when appropriate"
        "Answer strictly in plain text. Do NOT use Markdown, bullets, headers, or symbols like *, #, -."
        "Use light emojis only when it adds friendliness (e.g., 🙂,👍,🤗,✨,❤️,😊,😅)."
        "Keep the text readable, simple, and clean."
        "Sound like a helpful human, not a script. Be calm, clear, and kind in every interaction."
    )

    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    answer = openrouter_chat(system_instruction, prompt)

    return {"answer": answer}

