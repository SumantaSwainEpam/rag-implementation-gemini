import os
import pickle
from glob import glob
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY in .env or environment")

# initialize client (per Gemini docs)
client = genai.Client(api_key=API_KEY)

# choose the Gemini embedding model (example name; docs use gemini-embedding-001).
EMBED_MODEL = "gemini-embedding-001"

DOCS_DIR = "docs"
INDEX_FILE = "faiss_index.bin"
META_FILE = "docs_meta.pkl"

def read_documents(path):
    files = glob(os.path.join(path, "*.txt"))
    docs = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append({"path": p, "text": text})
    return docs

def embed_texts(texts):
    # call gemini embeddings endpoint as per docs
    # genai client returns embeddings in result.embeddings
    resp = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    # extract the actual embedding values from ContentEmbedding objects
    embeddings = []
    for embedding in resp.embeddings:
        if hasattr(embedding, 'values'):
            embeddings.append(embedding.values)
        else:
            # fallback if structure is different
            embeddings.append(list(embedding))
    return np.array(embeddings).astype("float32")

def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # use inner product on normalized vectors (cosine)
    # Normalize if using IP for cosine:
    faiss.normalize_L2(embs)
    index.add(embs)
    return index

def main():
    docs = read_documents(DOCS_DIR)
    texts = [d["text"] for d in docs]
    if not texts:
        print("No docs found in", DOCS_DIR)
        return

    print(f"Embedding {len(texts)} docs with model {EMBED_MODEL} ...")
    embs = embed_texts(texts)  # shape: (N, dim)

    print("Building FAISS index...")
    index = build_faiss_index(embs)

    print("Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(docs, f)

    print("Done. Index saved to", INDEX_FILE)

if __name__ == "__main__":
    main()
