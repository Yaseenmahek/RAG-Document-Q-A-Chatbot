"""
📄 RAG Document Q&A Chatbot
A Streamlit-based chatbot that lets you upload documents (PDF, TXT, DOCX, CSV)
and ask questions about their content using OpenAI's GPT with RAG.

Developed by Yaseen Mahek
"""

import streamlit as st
from config import (
    APP_TITLE, APP_DESCRIPTION, SIDEBAR_TITLE, 
    MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, AUTHOR_NAME
)
from rag_engine import (
    load_multiple_documents,
    split_into_chunks,
    create_vector_store,
    add_to_vector_store,
    get_streaming_response,
)


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A Chatbot | Yaseen Mahek",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ──────────────────────────────────────────────
# Premium Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* ===== Google Fonts ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ===== Global ===== */
    * { font-family: 'Inter', sans-serif !important; }
    code, pre { font-family: 'JetBrains Mono', monospace !important; }

    /* ===== Main App ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1035 40%, #0d1b2a 70%, #0a0a1a 100%);
        color: #e0e0f0;
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1140 50%, #0d1b2a 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.15);
    }

    /* ===== Chat Messages ===== */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(102, 126, 234, 0.12);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
    }

    /* ===== Chat Input ===== */
    [data-testid="stChatInput"] textarea {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(102, 126, 234, 0.25) !important;
        border-radius: 14px !important;
        color: #e0e0f0 !important;
        font-size: 15px !important;
        padding: 14px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.15) !important;
    }

    /* ===== Title Gradient ===== */
    .main-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        color: rgba(224, 224, 240, 0.6);
        font-size: 1.05rem;
        font-weight: 300;
        margin-bottom: 10px;
    }
    .author-tag {
        text-align: center;
        font-size: 0.85rem;
        color: rgba(102, 126, 234, 0.8);
        font-weight: 500;
        margin-bottom: 24px;
    }

    /* ===== Feature Cards ===== */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(102, 126, 234, 0.12);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        border-color: rgba(102, 126, 234, 0.35);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.12);
        background: rgba(102, 126, 234, 0.06);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
        display: block;
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e0f0;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: rgba(224, 224, 240, 0.5);
        line-height: 1.5;
    }

    /* ===== Welcome Box ===== */
    .welcome-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 32px;
        margin: 20px 0 30px 0;
        text-align: center;
    }
    .welcome-box h3 {
        color: #e0e0f0;
        font-weight: 700;
        margin-bottom: 16px;
    }
    .step-list {
        list-style: none;
        padding: 0;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 12px;
        margin: 0;
    }
    .step-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 0.95rem;
        color: rgba(224, 224, 240, 0.8);
        transition: all 0.3s ease;
    }
    .step-item:hover {
        border-color: rgba(102, 126, 234, 0.4);
        background: rgba(102, 126, 234, 0.08);
    }

    /* ===== Source References ===== */
    .source-ref {
        background: rgba(102, 126, 234, 0.06);
        border-left: 3px solid #667eea;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 12px 12px 0;
        font-size: 0.85rem;
        color: rgba(224, 224, 240, 0.7);
        transition: all 0.3s ease;
    }
    .source-ref:hover {
        background: rgba(102, 126, 234, 0.1);
        border-left-color: #764ba2;
    }

    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35);
    }
    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* ===== Upload Area ===== */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* ===== Metrics ===== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(102, 126, 234, 0.12);
        border-radius: 14px;
        padding: 16px;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(102, 126, 234, 0.3);
    }

    /* ===== Expander ===== */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 12px;
    }

    /* ===== Dividers ===== */
    hr {
        border-color: rgba(102, 126, 234, 0.1) !important;
        margin: 16px 0 !important;
    }

    /* ===== Status Badges ===== */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-success {
        background: rgba(0, 200, 83, 0.15);
        color: #00c853;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
    .badge-info {
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        padding: 24px 0 12px 0;
        color: rgba(224, 224, 240, 0.3);
        font-size: 0.85rem;
        border-top: 1px solid rgba(102, 126, 234, 0.08);
        margin-top: 40px;
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }

    /* ===== File Badge ===== */
    .file-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 8px;
        padding: 4px 10px;
        margin: 3px;
        font-size: 0.82rem;
        color: rgba(224, 224, 240, 0.7);
    }

    /* ===== Pulse animation for processing ===== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .processing { animation: pulse 1.5s ease-in-out infinite; }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
    ::-webkit-scrollbar-thumb { 
        background: rgba(102, 126, 234, 0.3); 
        border-radius: 3px; 
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(102, 126, 234, 0.5); }
</style>
""", unsafe_allow_html=True)





# ──────────────────────────────────────────────
# Initialize Session State
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    # Sidebar header
    st.markdown(f"""
    <div style="text-align: center; padding: 16px 0;">
        <div style="font-size: 2.5rem; margin-bottom: 4px;">📄</div>
        <div style="font-size: 1.1rem; font-weight: 700; 
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            RAG Chatbot
        </div>
        <div style="font-size: 0.75rem; color: rgba(224,224,240,0.4); margin-top: 2px;">
            by {AUTHOR_NAME}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Key Input
    st.markdown("##### 🔑 OpenAI API Key")
    api_key = st.text_input(
        "Enter your API key:",
        type="password",
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/api-keys",
        label_visibility="collapsed"
    )
    
    if api_key:
        st.markdown('<span class="status-badge badge-success">✓ API Key Set</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Enter your OpenAI API key to start")
    
    st.markdown("---")
    
    # File Upload
    st.markdown("##### 📁 Upload Documents")
    st.caption("Supports: PDF, TXT, DOCX, CSV")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        help=f"Max size: {MAX_FILE_SIZE_MB}MB per file",
        label_visibility="collapsed"
    )
    
    # Process Button
    if uploaded_files and api_key:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner(""):
                try:
                    # Progress indicators
                    progress_bar = st.progress(0, text="📖 Reading files...")
                    
                    # Step 1: Load documents
                    docs = load_multiple_documents(uploaded_files)
                    progress_bar.progress(33, text="✂️ Splitting into chunks...")
                    
                    if not docs:
                        st.error("❌ No text extracted from the files.")
                        progress_bar.empty()
                    else:
                        # Step 2: Chunk all documents
                        all_chunks = []
                        for doc in docs:
                            chunks = split_into_chunks(doc["text"], doc["filename"])
                            all_chunks.extend(chunks)
                        
                        progress_bar.progress(66, text="🧠 Creating embeddings...")
                        
                        # Step 3: Create or update vector store
                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = create_vector_store(all_chunks, api_key)
                        else:
                            st.session_state.vector_store = add_to_vector_store(
                                st.session_state.vector_store, all_chunks, api_key
                            )
                        
                        # Update state
                        new_files = [doc["filename"] for doc in docs]
                        st.session_state.processed_files.extend(new_files)
                        st.session_state.total_chunks += len(all_chunks)
                        
                        progress_bar.progress(100, text="✅ Done!")
                        st.success(f"Processed **{len(docs)} document(s)** — **{len(all_chunks)} chunks** created")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # Document Stats
    if st.session_state.processed_files:
        st.markdown("##### 📊 Loaded Documents")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Files", len(st.session_state.processed_files))
        with col2:
            st.metric("🧩 Chunks", st.session_state.total_chunks)
        
        # File badges
        file_badges = ""
        for fname in st.session_state.processed_files:
            ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else "FILE"
            icon = {"PDF": "📕", "TXT": "📝", "DOCX": "📘", "CSV": "📊"}.get(ext, "📄")
            file_badges += f'<span class="file-badge">{icon} {fname}</span>'
        st.markdown(file_badges, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.processed_files = []
            st.session_state.total_chunks = 0
            st.rerun()
    
    # Sidebar footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; font-size: 0.75rem; color: rgba(224,224,240,0.3); padding: 8px 0;">
        Powered by OpenAI • LangChain • FAISS<br>
        <span style="color: rgba(102,126,234,0.6);">Developed by {AUTHOR_NAME}</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="main-title">{APP_TITLE}</div>
<div class="sub-title">{APP_DESCRIPTION}</div>
<div class="author-tag">Developed by {AUTHOR_NAME}</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Welcome screen (when no documents loaded)
if not st.session_state.vector_store:
    # Welcome box
    st.markdown("""
    <div class="welcome-box">
        <h3>👋 Welcome! Get started in 4 easy steps</h3>
        <div class="step-list">
            <div class="step-item">1️⃣ Enter your OpenAI API Key</div>
            <div class="step-item">2️⃣ Upload documents (PDF, TXT, DOCX, CSV)</div>
            <div class="step-item">3️⃣ Click "Process Documents"</div>
            <div class="step-item">4️⃣ Ask anything about your docs!</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("📄", "Multi-Format", "Upload PDF, TXT, DOCX, and CSV files"),
        ("🧠", "Smart RAG", "AI retrieves the most relevant document sections"),
        ("⚡", "Streaming", "Real-time token-by-token responses"),
        ("💬", "Memory", "Follow-up questions with context"),
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <span class="feature-icon">{icon}</span>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How RAG works section
    with st.expander("🔍 How does RAG work?", expanded=False):
        st.markdown("""
        **Retrieval-Augmented Generation (RAG)** is a cutting-edge AI technique that grounds LLM responses in your actual documents:
        
        1. **📄 Document Processing** — Your files are parsed, split into chunks, and converted to vector embeddings
        2. **🔎 Semantic Search** — Your question is embedded and compared against document chunks using cosine similarity
        3. **🤖 Augmented Generation** — The most relevant chunks are injected into the prompt, ensuring accurate answers
        
        This means the AI answers from **your documents**, not from general knowledge — reducing hallucinations!
        """)


# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 Source References", expanded=False):
                for source in message["sources"]:
                    st.markdown(f"""
                    <div class="source-ref">
                        <strong>📄 {source['file']}</strong> &nbsp;|&nbsp; Chunk {source['chunk']}<br>
                        <em>"{source['preview']}"</em>
                    </div>
                    """, unsafe_allow_html=True)


# Chat Input
if prompt := st.chat_input("💬 Ask a question about your documents..."):
    # Validate state
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar first.")
    elif not st.session_state.vector_store:
        st.error("⚠️ Please upload and process at least one document first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            source_docs = None
            
            try:
                for token, docs in get_streaming_response(
                    query=prompt,
                    vector_store=st.session_state.vector_store,
                    chat_history=st.session_state.messages[:-1],
                    api_key=api_key
                ):
                    full_response += token
                    message_placeholder.markdown(full_response + "▌")
                    source_docs = docs
                
                message_placeholder.markdown(full_response)
                
                # Prepare source references
                sources = []
                if source_docs:
                    for doc, score in source_docs:
                        sources.append({
                            "file": doc.metadata.get("source", "Unknown"),
                            "chunk": doc.metadata.get("chunk_index", "?"),
                            "preview": doc.page_content[:150] + "..."
                        })
                    
                    with st.expander("📚 Source References", expanded=False):
                        for source in sources:
                            st.markdown(f"""
                            <div class="source-ref">
                                <strong>📄 {source['file']}</strong> &nbsp;|&nbsp; Chunk {source['chunk']}<br>
                                <em>"{source['preview']}"</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
            
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


# Footer
st.markdown(f"""
<div class="footer">
    📄 RAG Document Q&A Chatbot &nbsp;•&nbsp; Built with OpenAI, LangChain, FAISS & Streamlit<br>
    Developed by <strong>{AUTHOR_NAME}</strong>
</div>
""", unsafe_allow_html=True)
