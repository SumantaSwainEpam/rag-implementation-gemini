import streamlit as st
import os
from pathlib import Path
from ingest_docs import main as ingest_main, DOCS_DIR, INDEX_FILE, META_FILE
from rag_query import (
    load_index, embed_query, retrieve_topk, generate_answer,
    GEN_MODEL, EMBED_MODEL
)
import pickle

# Page configuration
st.set_page_config(
    page_title="RAG with Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'index_loaded' not in st.session_state:
    st.session_state.index_loaded = False
if 'index' not in st.session_state:
    st.session_state.index = None
if 'docs' not in st.session_state:
    st.session_state.docs = None

def load_index_if_exists():
    """Load the FAISS index if it exists."""
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        try:
            index, docs = load_index()
            st.session_state.index = index
            st.session_state.docs = docs
            st.session_state.index_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading index: {e}")
            return False
    return False

# Header
st.markdown('<div class="main-header">🤖 RAG Implementation with Gemini</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check index status
    index_exists = os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)
    
    if index_exists:
        st.success("✅ Index found")
        if st.button("🔄 Reload Index"):
            if load_index_if_exists():
                st.success("Index reloaded successfully!")
                st.rerun()
    else:
        st.warning("⚠️ No index found")
        st.info("Please ingest documents first to create an index.")
    
    st.divider()
    
    st.subheader("📊 Model Information")
    st.text(f"Embedding: {EMBED_MODEL}")
    st.text(f"Generation: {GEN_MODEL}")
    
    st.divider()
    
    st.subheader("📁 Document Directory")
    st.text(f"{DOCS_DIR}/")
    if os.path.exists(DOCS_DIR):
        files = list(Path(DOCS_DIR).glob("*"))
        txt_files = [f for f in files if f.suffix == '.txt']
        pdf_files = [f for f in files if f.suffix == '.pdf']
        st.text(f"📄 Text files: {len(txt_files)}")
        st.text(f"📕 PDF files: {len(pdf_files)}")
    else:
        st.warning(f"Directory '{DOCS_DIR}' not found")

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Query Documents", "📥 Ingest Documents", "ℹ️ About"])

with tab1:
    st.header("Ask Questions")
    
    # Load index if available
    if not st.session_state.index_loaded:
        if load_index_if_exists():
            st.session_state.index_loaded = True
    
    if not st.session_state.index_loaded:
        st.warning("⚠️ No index loaded. Please ingest documents first in the 'Ingest Documents' tab.")
        st.info("""
        **Steps to get started:**
        1. Place your `.txt` or `.pdf` files in the `docs/` folder
        2. Go to the "Ingest Documents" tab
        3. Click "Ingest Documents" to build the index
        4. Return here to ask questions
        """)
    else:
        # Query interface
        st.markdown("### Enter your question:")
        
        query = st.text_input(
            "Question",
            placeholder="e.g., What is Retrieval-Augmented Generation?",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            k_value = st.number_input("Top K results", min_value=1, max_value=10, value=3)
        with col2:
            st.write("")  # Spacing
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Processing your question..."):
                    try:
                        # Embed query
                        qvec = embed_query(query)
                        
                        # Retrieve top-k
                        scores, ids = retrieve_topk(st.session_state.index, qvec, k=k_value)
                        
                        # Get retrieved documents
                        retrieved_texts = []
                        retrieved_meta = []
                        for idx in ids:
                            if idx >= 0 and idx < len(st.session_state.docs):
                                meta = st.session_state.docs[idx]
                                retrieved_texts.append(
                                    f"FILE: {meta['path']}\n{meta['text'][:1000]}"
                                )
                                retrieved_meta.append({
                                    'score': float(scores[list(ids).index(idx)]),
                                    'path': meta['path'],
                                    'chunk': meta.get('chunk_index', 0),
                                    'total_chunks': meta.get('total_chunks', 1)
                                })
                        
                        # Display retrieved documents
                        st.markdown("### 📚 Retrieved Documents")
                        for i, meta in enumerate(retrieved_meta, 1):
                            with st.expander(f"📄 {Path(meta['path']).name} (Score: {meta['score']:.4f})"):
                                st.text(f"File: {meta['path']}")
                                if meta['total_chunks'] > 1:
                                    st.text(f"Chunk {meta['chunk'] + 1} of {meta['total_chunks']}")
                                st.text(f"Similarity Score: {meta['score']:.4f}")
                                st.text_area(
                                    "Content Preview",
                                    retrieved_texts[i-1],
                                    height=150,
                                    disabled=True,
                                    key=f"preview_{i}"
                                )
                        
                        # Generate answer
                        st.markdown("### 💡 Generated Answer")
                        with st.spinner("Generating answer using Gemini..."):
                            answer = generate_answer(query, retrieved_texts)
                            st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
                        st.exception(e)
            else:
                st.warning("Please enter a question.")

with tab2:
    st.header("Document Ingestion")
    
    st.markdown("""
    ### 📥 Ingest Documents
    
    This process will:
    1. Read all `.txt` and `.pdf` files from the `docs/` folder
    2. Split documents into chunks using recursive character splitting
    3. Generate embeddings using Gemini's embedding model
    4. Build a FAISS index for fast similarity search
    
    **Supported formats:** `.txt`, `.pdf`
    """)
    
    # Show current documents
    if os.path.exists(DOCS_DIR):
        files = list(Path(DOCS_DIR).glob("*"))
        txt_files = [f for f in files if f.suffix == '.txt']
        pdf_files = [f for f in files if f.suffix == '.pdf']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Text Files", len(txt_files))
        with col2:
            st.metric("PDF Files", len(pdf_files))
        
        if txt_files or pdf_files:
            st.markdown("#### Files found:")
            for f in sorted(txt_files + pdf_files):
                st.text(f"  • {f.name}")
        else:
            st.warning(f"No `.txt` or `.pdf` files found in `{DOCS_DIR}/`")
    else:
        st.warning(f"Directory `{DOCS_DIR}/` does not exist. Please create it and add your documents.")
    
    st.divider()
    
    if st.button("🚀 Ingest Documents", type="primary", use_container_width=True):
        with st.spinner("Ingesting documents... This may take a while."):
            try:
                # Capture output
                import sys
                from io import StringIO
                
                # Redirect stdout to capture print statements
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                # Run ingestion
                ingest_main()
                
                # Restore stdout
                sys.stdout = old_stdout
                output = captured_output.getvalue()
                
                # Display output
                st.success("✅ Document ingestion completed successfully!")
                st.code(output, language="text")
                
                # Reload index
                if load_index_if_exists():
                    st.session_state.index_loaded = True
                    st.success("Index loaded and ready for queries!")
                    st.rerun()
                
            except Exception as e:
                st.error(f"Error during ingestion: {e}")
                st.exception(e)

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ## 🤖 RAG Implementation with Gemini
    
    A complete Retrieval-Augmented Generation (RAG) system using Google's Gemini API.
    
    ### ✨ Features
    
    - **Document Ingestion**: Process and embed text and PDF documents
    - **Intelligent Chunking**: Recursive character splitting for better semantic boundaries
    - **Semantic Search**: FAISS vector similarity search
    - **Contextual Q&A**: Generate accurate answers using retrieved context
    - **Source Citation**: Automatically cite source documents
    - **Modern Web UI**: Beautiful Streamlit interface
    
    ### 🔧 Technical Details
    
    - **Embedding Model**: `gemini-embedding-001`
    - **Generation Model**: `gemini-2.5-flash`
    - **Vector Search**: FAISS with cosine similarity
    - **Chunking Strategy**: Recursive character splitting with overlap
    
    ### 📚 How It Works
    
    1. **Document Processing**: Documents are split into chunks with overlap
    2. **Embedding**: Each chunk is embedded using Gemini's embedding model
    3. **Indexing**: FAISS index is built for fast similarity search
    4. **Query**: User questions are embedded and matched against the index
    5. **Generation**: Top-k relevant chunks are used as context for answer generation
    
    ### 👤 Author
    
    **Sumanta Swain** (sumanta_swain@epam.com)
    
    ---
    
    **Happy RAG-ing! 🚀**
    """)

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built with ❤️ using Streamlit and Google Gemini API"
    "</div>",
    unsafe_allow_html=True
)
