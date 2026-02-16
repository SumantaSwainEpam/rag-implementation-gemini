import os
import pickle
from glob import glob
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

load_dotenv()
# Support both GOOGLE_API_KEY and GEMINI_API_KEY for convenience
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env or environment")

# initialize client (per Gemini docs)
client = genai.Client(api_key=API_KEY)

# choose the Gemini embedding model (example name; docs use gemini-embedding-001).
EMBED_MODEL = "gemini-embedding-001"

DOCS_DIR = "docs"
INDEX_FILE = "faiss_index.bin"
META_FILE = "docs_meta.pkl"

# Chunking configuration
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks

def recursive_split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Recursively split text into chunks with overlap.
    Uses recursive character splitting strategy for better semantic boundaries.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 20% of the chunk
            search_start = max(start, end - int(chunk_size * 0.2))
            for i in range(end, search_start, -1):
                if text[i-1:i+1] in ['.\n', '.\r', '.\r\n', '.\t', '. ']:
                    end = i
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks

def read_documents(path):
    """
    Read documents from the specified path.
    Supports .txt and .pdf files.
    """
    docs = []
    
    # Read text files
    txt_files = glob(os.path.join(path, "*.txt"))
    for p in txt_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append({"path": p, "text": text, "type": "txt"})
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
    
    # Read PDF files
    try:
        import PyPDF2
        pdf_files = glob(os.path.join(path, "*.pdf"))
        for p in pdf_files:
            try:
                text = ""
                with open(p, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                text = text.strip()
                if text:
                    docs.append({"path": p, "text": text, "type": "pdf"})
            except Exception as e:
                print(f"Warning: Could not read PDF {p}: {e}")
    except ImportError:
        print("Warning: PyPDF2 not installed. PDF files will be skipped.")
        print("Install with: uv add PyPDF2")
    
    return docs

def chunk_documents(docs):
    """
    Split documents into chunks using recursive character splitting.
    """
    chunked_docs = []
    for doc in docs:
        chunks = recursive_split_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "path": doc["path"],
                "text": chunk,
                "type": doc.get("type", "txt"),
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
    return chunked_docs

def embed_texts(texts):
    # Call Gemini embeddings endpoint and normalize output to list of float vectors
    resp = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    vectors = []
    for emb in resp.embeddings:
        # SDKs often return ContentEmbedding with `.values`; fallback to iterable
        if hasattr(emb, "values"):
            vec = np.array(emb.values, dtype="float32")
        else:
            vec = np.array(list(emb), dtype="float32")
        vectors.append(vec)
    return np.vstack(vectors)

def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # use inner product on normalized vectors (cosine)
    # Normalize if using IP for cosine:
    faiss.normalize_L2(embs)
    index.add(embs)
    return index

def main():
    docs = read_documents(DOCS_DIR)
    if not docs:
        print(f"No documents found in {DOCS_DIR}")
        print("Supported formats: .txt, .pdf")
        return

    print(f"Found {len(docs)} document(s)")
    
    # Chunk documents for better retrieval
    print("Chunking documents...")
    chunked_docs = chunk_documents(docs)
    print(f"Created {len(chunked_docs)} chunks from {len(docs)} document(s)")
    
    texts = [d["text"] for d in chunked_docs]
    if not texts:
        print("No text content found in documents")
        return

    print(f"Embedding {len(texts)} chunks with model {EMBED_MODEL} ...")
    embs = embed_texts(texts)  # shape: (N, dim)

    print("Building FAISS index...")
    index = build_faiss_index(embs)

    print("Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(chunked_docs, f)

    print(f"Done. Index saved to {INDEX_FILE}")
    print(f"Total chunks indexed: {len(chunked_docs)}")

if __name__ == "__main__":
    main()
