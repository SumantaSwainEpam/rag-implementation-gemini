import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

load_dotenv()
# Support both GOOGLE_API_KEY and GEMINI_API_KEY for convenience
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env or environment")

client = genai.Client(api_key=API_KEY)
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"   # example text generation model; pick one available to you

INDEX_FILE = "faiss_index.bin"
META_FILE = "docs_meta.pkl"

def embed_query(q):
    resp = client.models.embed_content(model=EMBED_MODEL, contents=[q])
    # extract the actual embedding values from ContentEmbedding object
    if hasattr(resp.embeddings[0], 'values'):
        vec = np.array(resp.embeddings[0].values, dtype="float32")
    else:
        # fallback if structure is different
        vec = np.array(list(resp.embeddings[0]), dtype="float32")
    # normalize for cosine (since index used normalized vectors)
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise SystemExit("Run ingest_docs.py first to build index.")
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        docs = pickle.load(f)
    return index, docs

def retrieve_topk(index, qvec, k=3):
    # qvec shape (dim,)
    q = qvec.reshape(1, -1)
    faiss.normalize_L2(q)  # ensure normalized
    scores, ids = index.search(q, k)
    return scores[0], ids[0]

def generate_answer(query, retrieved_texts):
    # Build a prompt that includes retrieved docs as context (short)
    context = "\n\n---\n\n".join(retrieved_texts)
    prompt = (
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer concisely and cite which context file you used."
    )

    # call Gemini text generation (per docs)
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt
    )
    # many SDKs have response.text or response.output; check your SDK return structure
    answer = getattr(response, "text", None) or response.output[0].content[0].text
    return answer

def main():
    index, docs = load_index()
    question = input("Enter your question: ").strip()
    qvec = embed_query(question)
    scores, ids = retrieve_topk(index, qvec, k=3)

    retrieved_texts = []
    for idx in ids:
        if idx < 0 or idx >= len(docs): 
            continue
        meta = docs[idx]
        retrieved_texts.append(f"FILE: {meta['path']}\n{meta['text'][:1000]}")  # limited preview

    print("\nRetrieved top documents (score, path):")
    for s, i in zip(scores, ids):
        if i >= 0 and i < len(docs):
            print(f"{s:.4f}  {docs[i]['path']}")

    print("\nGenerating answer using Gemini...")
    answer = generate_answer(question, retrieved_texts)
    print("\n=== Answer ===\n")
    print(answer)

if __name__ == "__main__":
    main()
