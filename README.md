# 📄 RAG Document Q&A Chatbot

An AI-powered chatbot that lets you **upload PDF documents** and **ask questions** about their content using Retrieval-Augmented Generation (RAG).

Built with **OpenAI GPT**, **LangChain**, **FAISS**, and **Streamlit**.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📁 Multi-PDF Upload | Upload and query multiple PDF documents at once |
| 🧠 RAG Pipeline | Intelligent retrieval of relevant document sections |
| 💬 Streaming Chat | Real-time token-by-token response streaming |
| 📚 Source References | See exactly which document sections the AI used |
| 🔄 Conversation Memory | Ask follow-up questions with full context |
| 🎨 Modern UI | Beautiful gradient dark theme with animations |

---

## 🏗️ Architecture

```
User uploads PDF → Extract text → Split into chunks → Generate embeddings
                                                            ↓
User asks question → Embed question → Retrieve relevant chunks from FAISS
                                                            ↓
                          Build prompt with context → Send to OpenAI GPT → Stream response
```

**Tech Stack:**
- **LLM:** OpenAI GPT-3.5-turbo / GPT-4
- **Embeddings:** OpenAI text-embedding-ada-002
- **Vector Store:** FAISS (local, free)
- **Orchestration:** LangChain
- **PDF Parsing:** PyPDF2
- **Frontend:** Streamlit

---

## 📂 Project Structure

```
rag-document-qa-chatbot/
├── app.py              # Main Streamlit application
├── rag_engine.py       # RAG pipeline (load, chunk, embed, retrieve, answer)
├── config.py           # Configuration & constants
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/rag-document-qa-chatbot.git
   cd rag-document-qa-chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** at `http://localhost:8501`

---

## 📖 Usage

1. **Enter your OpenAI API key** in the sidebar
2. **Upload one or more PDF documents** using the file uploader
3. **Click "Process Documents"** to analyze and embed the content
4. **Ask questions** in the chat input — the AI will answer based on your documents!
5. **View source references** by expanding the "📚 View Source References" section

---

## ☁️ Deploy on Streamlit Cloud

1. Push the code to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py` as the main file
4. Add your `OPENAI_API_KEY` in **Secrets Management**:
   ```toml
   OPENAI_API_KEY = "your_key_here"
   ```
5. Click **Deploy!**

---

## 🛠️ Configuration

Modify `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K_RESULTS` | 4 | Number of relevant chunks retrieved |
| `CHAT_MODEL` | gpt-3.5-turbo | OpenAI chat model |
| `TEMPERATURE` | 0.3 | Response creativity (0 = factual, 1 = creative) |

---

## 📝 How RAG Works

**Retrieval-Augmented Generation (RAG)** enhances LLMs by grounding their responses in your actual document content:

1. **Document Processing:** PDFs are parsed, split into chunks, and converted to vector embeddings
2. **Semantic Search:** When you ask a question, it's embedded and compared against document chunks using cosine similarity
3. **Augmented Generation:** The most relevant chunks are injected into the prompt, ensuring the AI answers based on your documents — not hallucinations

---

## ⚠️ Important Notes

- You need a **valid OpenAI API key** with billing enabled
- API costs are minimal (~$0.01-0.05 per query with GPT-3.5-turbo)
- Uploaded documents are processed **in-memory only** — nothing is stored permanently
- For large documents (100+ pages), processing may take 30-60 seconds

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Built with ❤️ using OpenAI, LangChain, FAISS & Streamlit**
