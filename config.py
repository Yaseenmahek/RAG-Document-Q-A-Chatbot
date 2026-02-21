"""
Configuration and constants for the RAG Document Q&A Chatbot.
"""

# --- Document Processing ---
CHUNK_SIZE = 1000          # Number of characters per chunk
CHUNK_OVERLAP = 200        # Overlap between consecutive chunks
SEPARATOR = "\n"           # Primary separator for text splitting

# --- Retrieval ---
TOP_K_RESULTS = 4          # Number of relevant chunks to retrieve

# --- OpenAI Models ---
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.3          # Lower = more factual, higher = more creative

# --- System Prompt ---
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided document context.

Rules:
1. Answer ONLY based on the provided context. Do not use your general knowledge.
2. If the answer is not found in the context, say: "I couldn't find the answer in the uploaded document(s). Please try rephrasing your question."
3. Be concise but thorough in your answers.
4. When possible, quote relevant parts from the document to support your answer.
5. If the question is a greeting or casual conversation, respond naturally but remind the user to ask about their documents.
"""

# --- UI Configuration ---
APP_TITLE = "📄 RAG Document Q&A Chatbot"
APP_DESCRIPTION = "Upload your documents and ask questions — powered by AI!"
SIDEBAR_TITLE = "⚙️ Configuration"
AUTHOR_NAME = "Yaseen Mahek"

# --- File Constraints ---
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = ["pdf", "txt", "docx", "csv"]
