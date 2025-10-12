# RAG Implementation with Gemini

A complete Retrieval-Augmented Generation (RAG) system using Google's Gemini API for embeddings and text generation, with FAISS for vector similarity search.

**Author**: Sumanta Swain (sumanta_swain@epam.com)

## 🚀 Features

- **Document Ingestion**: Process and embed text documents using Gemini's embedding model
- **Semantic Search**: Find relevant documents using FAISS vector similarity search
- **Contextual Q&A**: Generate accurate answers using retrieved context with Gemini
- **Source Citation**: Automatically cite source documents in responses
- **Interactive CLI**: Easy-to-use command-line interface

## 📋 Prerequisites

- Python 3.13+
- Google API Key (get from [Google AI Studio](https://aistudio.google.com/app/apikey))
- `uv` package manager

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd rag-implementation-gemini
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up your API key**:
   
   **Option A: Environment variable (recommended)**
   ```bash
   # Windows PowerShell
   $env:GOOGLE_API_KEY="your_api_key_here"
   
   # Linux/Mac
   export GOOGLE_API_KEY="your_api_key_here"
   ```
   
   **Option B: Create .env file**
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## 🚀 Usage

### Interactive Mode

Run the main application:
```bash
uv run python main.py
```

You'll see a menu:
```
=== RAG with Gemini ===
1. Ingest documents
2. Ask a question
0. Exit
```

### Step-by-Step Process

1. **Ingest Documents** (Option 1):
   - Place your `.txt` files in the `docs/` folder
   - Select option 1 to process and embed documents
   - Creates FAISS index and metadata files

2. **Ask Questions** (Option 2):
   - Select option 2 to query your documents
   - Enter your question
   - Get contextual answers with source citations

### Test the System

Run the automated test suite:
```bash
uv run python test_rag.py
```

This will test the system with sample questions and show retrieval scores and generated answers.

## 📁 Project Structure

```
rag-implementation-gemini/
├── docs/                    # Place your .txt documents here
│   ├── doc1.txt            # Sample AI/ML content
│   └── doc2.txt            # Sample RAG content
├── main.py                 # Main application entry point
├── ingest_docs.py          # Document ingestion and embedding
├── rag_query.py            # Query processing and answer generation
├── test_rag.py             # Automated test suite
├── pyproject.toml          # Project dependencies
├── README.md               # This file
├── faiss_index.bin         # Generated FAISS index (after ingestion)
└── docs_meta.pkl           # Generated document metadata (after ingestion)
```

## 🔧 Configuration

### Models Used

- **Embedding Model**: `gemini-embedding-001`
- **Text Generation Model**: `gemini-2.5-flash`
- **Vector Search**: FAISS with cosine similarity

### Customization

You can modify these settings in the respective files:

**In `ingest_docs.py` and `rag_query.py`**:
```python
EMBED_MODEL = "gemini-embedding-001"  # Change embedding model
GEN_MODEL = "gemini-2.5-flash"        # Change generation model
```

**In `rag_query.py`**:
```python
def retrieve_topk(index, qvec, k=3):  # Change number of retrieved docs
```

## 📊 How It Works

1. **Document Processing**:
   - Reads all `.txt` files from `docs/` folder
   - Generates embeddings using Gemini's embedding model
   - Creates FAISS index for fast similarity search

2. **Query Processing**:
   - Embeds the user's question
   - Searches for similar document chunks
   - Retrieves top-k most relevant passages

3. **Answer Generation**:
   - Combines retrieved context with the question
   - Sends to Gemini for contextual answer generation
   - Returns answer with source citations

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run automated tests
uv run python test_rag.py

# Test specific functionality
uv run python -c "from rag_query import load_index; print('Index loaded successfully')"
```

## 📝 Sample Questions

Try these questions with the included sample documents:

- "What is machine learning?"
- "How does RAG work?"
- "What are the benefits of RAG?"
- "What frameworks are mentioned for machine learning?"

## 🔍 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   Set GOOGLE_API_KEY in .env or environment
   ```
   **Solution**: Ensure your API key is properly set

2. **Empty Documents Error**:
   ```
   The text content is empty
   ```
   **Solution**: Add content to your `.txt` files in the `docs/` folder

3. **Import Errors**:
   ```
   Import "faiss" could not be resolved
   ```
   **Solution**: Run `uv sync` to install dependencies

### Performance Tips

- Use smaller document chunks for better retrieval accuracy
- Adjust the number of retrieved documents (k) based on your needs
- Consider using different embedding models for domain-specific content

## 🛠️ Development

### Adding New Document Types

To support other file formats, modify `read_documents()` in `ingest_docs.py`:

```python
def read_documents(path):
    files = glob(os.path.join(path, "*.txt"))  # Add more patterns
    # Add support for .pdf, .docx, etc.
```

### Customizing Retrieval

Modify the retrieval logic in `rag_query.py`:

```python
def retrieve_topk(index, qvec, k=3):
    # Adjust similarity threshold
    # Add filtering logic
    # Modify scoring
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue in the repository.

---

**Happy RAG-ing! 🚀**
