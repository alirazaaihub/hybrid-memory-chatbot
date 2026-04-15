# 🧠 Hybrid Memory Chatbot

A conversational AI chatbot with **two-layer memory** — Short-Term Memory (conversation summarization) and Long-Term Memory (RAG-based vector storage) — that remembers important user information across sessions.

---

## 🚀 What It Does

Most chatbots have no memory. This one has two layers:

- **Short-Term Memory** — keeps recent conversation, summarizes older messages automatically
- **Long-Term Memory** — extracts important user information and stores it in a vector database, retrieved on future queries

**Example:**
> "I am a machine learning student from Lahore" → stored in long-term memory  
> Come back tomorrow → bot still knows who you are  
> "I prefer concise answers" → remembered and applied in future responses

---

## ✨ Features

- 💬 **Short-Term Memory** — sliding window of last 5 messages, older ones auto-summarized
- 🧠 **Long-Term Memory** — LLM extracts key facts from user messages, stored in ChromaDB
- 🔍 **RAG-Based Recall** — retrieves relevant past memories using semantic similarity search
- 👤 **Per-User Memory** — memories are stored and filtered by `user_id`
- 🔁 **Duplicate Detection** — avoids storing the same memory twice
- ⚡ **Fast Inference** — powered by Groq (llama-3.1-8b-instant)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Short-Term Memory | Custom Summarizer Node |
| Long-Term Memory | LLM Extraction + ChromaDB |
| Language | Python 3.11+ |

---

## ⚙️ How It Works

```
User Message
      ↓
┌──────────────────────┐
│  Short-Term Memory   │  → Keep last 5 messages, summarize older ones
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Long-Term Retrieval │  → Search ChromaDB for relevant past memories
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   LLM Generation     │  → Answer using STM context + LTM context
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Memory Extraction   │  → Extract key facts from user message → save to ChromaDB
└──────────┬───────────┘
           ↓
        Response
```

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/alirazaaihub/hybrid-memory-chatbot
cd hybrid-memory-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Open `chatbot.py` and replace the API key:

```python
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your free Groq API key at: https://console.groq.com

---

## ▶️ Usage

```bash
python chatbot.py
```

**Example conversation:**

```
You: My name is Ali and I am learning AI from YouTube
AI: Nice to meet you Ali! That's great that you're learning AI...

You: What topics should I study next?
AI: Based on what you've shared Ali, I'd recommend...

You: exit
```

Memory is saved automatically. Next time you run the chatbot with the same `user_id`, it will remember Ali and his background.

---

## 📁 Project Structure

```
hybrid-memory-chatbot/
│
├── chatbot.py                  # Main chatbot logic
├── requirements.txt            # Project dependencies
├── db/
│   └── chroma_user_memory/     # ChromaDB long-term memory store (auto-created)
└── README.md
```

---

## 📋 Dependencies

```
langchain
langchain-groq
langchain-huggingface
langchain-chroma
langchain-core
sentence-transformers
chromadb
```

Install all:

```bash
pip install langchain langchain-groq langchain-huggingface langchain-chroma langchain-core sentence-transformers chromadb
```

---

## 🧩 Key Concepts Used

| Concept | How It's Used |
|--------|----------------|
| **Short-Term Memory** | Keeps last 5 messages; older ones summarized by LLM |
| **Long-Term Memory** | LLM extracts facts from user input → stored in ChromaDB |
| **RAG Retrieval** | Semantic similarity search over stored memories |
| **Per-User Filtering** | Each user's memory stored with `user_id` metadata |
| **Duplicate Detection** | Checks similarity before saving to avoid redundant entries |
| **Conversation Summarization** | Older messages compressed to stay within token limits |

---

## ⚙️ Configuration

You can tweak these values in `chatbot.py`:

```python
MAX_SHORT_TERM = 5      # How many recent messages to keep before summarizing
MAX_TOKENS = 2000       # Max token limit for conversation summary
TOP_K = 5               # How many long-term memories to retrieve per query
```

---

## 🙋 About

Built by **Ali raza**-old self-taught AI developer from Pakistan.  
This is part of my Agentic AI portfolio built using LangChain and Groq.

📌 [LinkedIn](https://www.linkedin.com/in/ali-raza-7124a0403/) • [GitHub](https://github.com/alirazaaihub)

---

## 📄 License

MIT License — feel free to use and modify.
