🧠 RAG Chatbot for Health & Wellbeing

A Retrieval-Augmented Generation (RAG) chatbot designed for a non-profit health and wellbeing platform. It provides accurate, document-grounded responses using a curated knowledge base of trusted sources.

🚀 Features
🔍 Semantic search over internal documents using vector embeddings
📚 Answers strictly based on retrieved context (no hallucinations)
⚠️ Graceful fallback when information is not available
🌐 Deployable backend with FastAPI
💬 Lightweight frontend chat widget (HTML + JS)
🔒 API key secured on backend (not exposed to users)
🏗️ Tech Stack
Backend: FastAPI
Vector Database: ChromaDB
Embeddings: BGE / OpenRouter embeddings
LLM: OpenRouter (LLaMA / other models)
Frontend: HTML, CSS, JavaScript
📁 Project Structure
backend/
│
├── documents/        # Source knowledge base (text files)
├── rag_db/           # Chroma vector database
├── main.py           # FastAPI app
├── rebuild_db.py     # Script to rebuild vector DB
├── requirements.txt
│
index.html            # Chat widget frontend
⚙️ Setup Instructions
1. Clone the repository
git clone <your-repo-url>
cd AIchatbot/backend
2. Install dependencies
pip install -r requirements.txt
3. Set environment variables

Create a .env file:

OPENROUTER_KEY=your_api_key_here
🧠 Build the Vector Database
python rebuild_db.py

This will:

Read all files from documents/
Chunk text
Generate embeddings
Store them in rag_db/

 Run the Backend
uvicorn main:app --reload

Server runs

 API Endpoint
POST /chat

Request:

{
  "question": "What is metabolic therapy?"
}

Response:

{
  "answer": "..."
}
💬 Frontend Usage

Update your frontend fetch URL:

const BACKEND_URL = "https://your-backend.onrender.com/chat";

Open index.html in browser to test.

☁️ Deployment
Backend 
Build command:
pip install -r requirements.txt
Start command:
uvicorn main:app --host 0.0.0.0 --port $PORT

Frontend

Deploy on:

GitHub 
Vercel
Netlify

⚠️ Notes
Ensure embedding model matches the one used to build rag_db
Do NOT expose API keys in frontend
Large embedding models may exceed free-tier memory limits

🎯 Goal
To provide safe, transparent, and reliable access to health and wellbeing information while minimizing misinformation.

📜 License

For non-profit and educational use.



For non-profit and educational use.

