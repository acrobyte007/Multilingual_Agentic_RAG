# 🌍 Multilingual RAG System (English, Hindi, Bengali)

A powerful **Retrieval-Augmented Generation (RAG)** system designed for **multilingual document search and question answering** across **English, Hindi (हिन्दी), and Bengali (বাংলা)**.

This system enables **cross-lingual retrieval**, meaning users can query in one language and retrieve relevant information from documents in another — all while receiving responses in their original query language.

---

## 🚀 Features

### 🌐 Multilingual Capabilities

* ✅ Full support for:

  * Hindi (Devanagari script)
  * Bengali (বাংলা script)
  * English (native support)
* 🔄 Cross-lingual retrieval:

  * Query in one language, retrieve documents in another
* 🧠 Language-aware ranking:

  * Custom BM25 reranking with language-specific tokenization
* 💬 Language-preserving responses:

  * Answers are generated in the user’s input language

---

## 🧩 Core Capabilities

* 📄 Document ingestion from:

  * PDF
  * DOCX
  * DOC (via LibreOffice)
* ✂️ Intelligent text chunking with overlap
* 🔍 Hybrid search:

  * Semantic vector search + keyword-based retrieval
* 🧠 Multilingual embeddings (E5 model)
* 📊 BM25 reranking for improved relevance
* 🤖 LLM-powered answer generation with source citations

---

## 🏗️ System Architecture

### 1. Document Processing Pipeline

#### 📥 Text Extraction

* PDFs → `pdfplumber`
* DOCX/DOC → `python-docx` (+ LibreOffice for `.doc`)

#### 🧹 Text Cleaning

* Unicode normalization
* Special character removal
* Whitespace cleanup

#### ✂️ Intelligent Chunking

* Sentence-aware splitting
* Default chunk size: **150 words**
* Default overlap: **20 words**

---

### 2. Embedding Layer

* **Model:** `gemini-embedding-001`

---

### 3. Vector Database

* **Platform:** Pinecone (serverless)

* **Similarity Metric:** Cosine similarity

* **Metadata Stored:**

  * Chunk text
  * Language
  * Tokens
  * Document ID

* **Filtering Support:**

  * Document-level filtering during retrieval

---

### 4. Search & Retrieval Pipeline

#### 🔎 Step 1: Semantic Search

* Vector similarity using embeddings

#### 🔁 Step 2: BM25 Reranking

* Language-aware tokenization
* IDF-based term weighting
* Score normalization (min-max scaling)

#### 🔀 Hybrid Retrieval

* Combines semantic understanding with keyword precision

---

### 5. Language Handling

* 🏷️ Language detection using `langdetect`
* 🔤 Language-specific token filtering
* 🌍 Cross-lingual query support
* 🔁 Query translation (Hindi ↔ Bengali ↔ English)

---

### 6. LLM Integration

* **Model:** Mistral AI (`ministral-8b-latest`)

---

## ⚙️ Local Setup

### 📋 Prerequisites

* Python **3.12+**
* LibreOffice (required for `.doc` files)
* Pinecone account (vector DB)
* Mistral AI API key

---

### 🛠️ Installation

```bash
git clone https://github.com/acrobyte007/Multilingual_Agentic_RAG
cd multilingual-rag

pip install -r requirements.txt
```

---

### 🔑 Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY
GOOGLE_API_KEY
MISTRAL_API_KEY
```

---

### ▶️ Run the System

```bash
python main.py
```

---

## 📊 Example Workflow

1. Upload documents (PDF/DOCX/DOC)
2. System processes and chunks text
3. Embeddings are generated and stored in Pinecone
4. User submits a query (any supported language)
5. System:

   * Detects language
   * Retrieves relevant chunks
   * Applies BM25 reranking
6. LLM generates a response with citations

---

## 🧪 Example Queries

* **English:**
  *"What are the key points in this report?"*

* **Hindi:**
  *"इस दस्तावेज़ का सारांश क्या है?"*

* **Bengali:**
  *"এই নথির মূল বিষয় কী?"*

---

## 📈 Future Improvements

* 🔊 Voice-based multilingual queries
* 📱 Web UI / dashboard
* 🧠 Fine-tuned domain-specific embeddings
* 📚 Support for more languages
* ⚡ Faster indexing with batch pipelines

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

* Multilingual E5 embeddings
* Pinecone vector database
* Mistral AI for LLM capabilities

---
