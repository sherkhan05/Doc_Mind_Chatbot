# 📄 DocMind – Local Document Intelligence Chatbot

DocMind is a **local Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload documents and chat with them intelligently.
The system processes documents, converts them into embeddings, stores them in a vector database, and retrieves relevant information to answer user queries.

Built with **Streamlit, Qdrant, and local embeddings**, the project enables **private, fast, and efficient document question answering** without relying on external APIs.

---

# 🚀 Features

✅ Upload and analyze multiple documents

✅ Intelligent document Q&A using **RAG (Retrieval-Augmented Generation)**

✅ Automatic document chunking and embedding generation

✅ Fast semantic search using **Qdrant Vector Database**

✅ Interactive chat interface built with **Streamlit**

✅ Source citation with answers

✅ Docker support for easy deployment

✅ Fully local processing for **privacy and security**

---

# 🧠 How It Works

The system follows a **RAG pipeline**:

1️⃣ User uploads documents

2️⃣ Documents are **split into chunks**

3️⃣ Chunks are converted into **vector embeddings**

4️⃣ Embeddings are stored in **Qdrant vector database**

5️⃣ User asks a question

6️⃣ System retrieves the **most relevant chunks**

7️⃣ The model generates an answer using retrieved context


# 🏗 Project Structure

```
Doc_Chatbot/
│
├── app/
│   ├── main.py              # Streamlit application entry point
│   ├── components.py       # UI components
│   ├── config.py           # Application configuration
│
├── core/
│   ├── document_loader.py  # Document ingestion & parsing
│   ├── embeddings.py       # Embedding model logic
│   ├── metadata.py         # Document metadata handling
│   ├── rag_pipeline.py     # RAG pipeline implementation
│   ├── utils.py            # Helper utilities
│   └── vectorstore.py      # Qdrant vector database integration
│
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container deployment
├── requirements.txt        # Python dependencies
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/docmind-chatbot.git
cd docmind-chatbot
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app/main.py
```

Then open your browser at:

```
http://localhost:8501
```

---

# 🐳 Run with Docker (Recommended)

Build and run the project using Docker:

```bash
docker-compose up --build
```

This will start:

* The Streamlit application
* Qdrant vector database

---

# 📂 Supported Workflow

1. Upload document(s)
2. System processes and indexes them
3. Ask questions about the documents
4. Get contextual answers with citations

---

# 📊 Key Technologies

* **Python**
* **Streamlit**
* **Qdrant Vector Database**
* **Embedding Models**
* **RAG (Retrieval Augmented Generation)**
* **Docker**

---

# 🔐 Privacy Advantage

Unlike cloud AI tools, this system can run **completely locally**, ensuring:

* Data privacy
* No API cost
* Secure document processing

---

# 🧪 Future Improvements

* Multi-document collections
* PDF / Word / PowerPoint parsing improvements
* Conversation memory
* Hybrid search (keyword + semantic)
* Authentication system
* Deployment to cloud platforms



