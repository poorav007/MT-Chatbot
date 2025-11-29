# ingest.py
import os
import glob
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

# CONFIG
DATA_DIR = "documents"
DB_DIR = "rag_db"
CHUNK_SIZE = 250      # words per chunk (changed from 800 characters)
CHUNK_OVERLAP = 50    # overlapping words (changed from 150 characters)

# load local embedding model
print("Loading embedding model (local)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    # returns list of lists (vectors)
    vecs = embed_model.encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]

def read_documents():
    docs = []
    filenames = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        name = os.path.basename(path)
        if name.startswith("."):
            continue
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            docs.append(text)
            filenames.append(name)
        elif path.lower().endswith(".pdf"):
            text = ""
            reader = PdfReader(path)
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
            docs.append(text.strip())
            filenames.append(name)
    return filenames, docs

def chunk_text(text, chunk_size=250, overlap=50):
    """
    Chunk text by words with overlap.
    chunk_size: target words per chunk (default: 250)
    overlap: overlapping words (default: 50)
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        
        # Only add non-empty chunks with meaningful content
        if chunk and len(chunk_words) >= 10:  # Skip tiny chunks
            chunks.append(chunk)
        
        start += (chunk_size - overlap)
        
        if end >= len(words):
            break
    
    return chunks

# --- main ingestion ---
if __name__ == "__main__":
    os.makedirs(DB_DIR, exist_ok=True)
    print("Reading documents from:", DATA_DIR)
    filenames, docs = read_documents()
    if not docs:
        print("No documents found in", DATA_DIR)
        raise SystemExit(1)

    print(f"Found {len(docs)} documents, now chunking...")
    all_chunks = []
    metadatas = []
    ids = []
    for i, (name, text) in enumerate(zip(filenames, docs)):
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    
    # Clean chunks before adding
    for j, ch in enumerate(chunks):
        # Additional cleaning: remove excessive whitespace
        ch = " ".join(ch.split())
        
        # Skip if chunk is too short or empty after cleaning
        if not ch or len(ch.split()) < 10:
            continue
            
        cid = f"{i}_{j}"
        all_chunks.append(ch)
        metadatas.append({"source": name, "chunk": j})
        ids.append(cid)

    print(f"Total chunks: {len(all_chunks)} — creating embeddings (local)...")
    embeddings = embed_texts(all_chunks)

    print("Starting Chroma client and storing vectors...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # create or get collection; we add embeddings directly so no embedding_function needed
    collection = client.get_or_create_collection(name="docs")

    # remove any existing docs with same ids (safe re-ingest)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    print("✅ Ingest complete. Chroma DB saved to", DB_DIR)


