# chatbot.py
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
print("API key loaded:", OPENROUTER_KEY)

if not OPENROUTER_KEY:
    raise SystemExit("Set OPENROUTER_KEY in .env")

# Database + Embedding Model
DB_DIR = "rag_db"
TOP_K = 3
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection("docs")

# Casual responses (NO RAG)
CASUAL_PHRASES = [
    "hi", "hello", "hey", "yo", "sup", "bye",
    "good morning", "good night", "what's up"
]

def embed_query(text):
    v = embed_model.encode([text], show_progress_bar=False)[0]
    return v.tolist()

def retrieve_context(query, k=TOP_K):
    q_vec = embed_query(query)
    res = collection.query(query_embeddings=[q_vec], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    combined = []
    for d, m in zip(docs, metas):
        src = m.get("source", "unknown")
        chunk_idx = m.get("chunk", None)
        combined.append(f"Source: {src} (chunk {chunk_idx})\n{d}")

    return "\n\n---\n\n".join(combined)

def openrouter_chat(system_instruction, user_prompt):
    """Calls OpenRouter but returns ONLY the model output (no debug)."""
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "MyRAGApp"
    }

    payload = {
        "model":"x-ai/grok-4.1-fast",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }

    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    j = r.json()

    return j["choices"][0]["message"]["content"]

def answer_query(query):
    # 1. CASUAL CHAT (no RAG)
    if query.lower().strip() in CASUAL_PHRASES:
        return "Hello! How can I help you today?"

    # 2. STRICT RAG for everything else
    context = retrieve_context(query)

    if not context.strip():
        return "Not enough information in the documents."

    system_instruction = (
        "You must answer ONLY using the provided CONTEXT. "
        "If the answer is not found, reply exactly: 'Not enough information in the documents.' "
        "Do NOT use outside knowledge."
    )

    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    answer = openrouter_chat(system_instruction, prompt)
    return answer

if __name__ == "__main__":
    print("RAG bot ready. Type a question (or 'exit')\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("\nBot:")
        try:
            print(answer_query(q))
        except Exception as e:
            print("Error:", e)
