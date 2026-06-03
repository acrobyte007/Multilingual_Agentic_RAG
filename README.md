# RAG System with Multilingual Support

A powerful Retrieval-Augmented Generation (RAG) system that supports multilingual document search and retrieval across English, Hindi, and Bengali languages.

## Features

### 🌐 Multilingual Support
- **Hindi (हिन्दी)** - Full support for Devanagari script
- **Bengali (বাংলা)** - Full support for Bengali script  
- **English** - Native language support
- **Cross-lingual retrieval** - Search in one language, find relevant documents in another
- **Language-aware BM25 reranking** - Uses language-specific tokenization for better relevance

### 🚀 Core Capabilities
- Document ingestion from PDF, DOCX, and DOC formats
- Intelligent text chunking with configurable overlap
- Semantic vector embeddings using multilingual E5 model
- Hybrid search combining semantic and keyword-based retrieval
- BM25 reranking for improved result relevance
- LLM-powered response generation with source citation

## Techniques Used

### 1. **Document Processing Pipeline**
- **Text Extraction**: PDFPlumber for PDFs, python-docx for DOCX/DOC files
- **Text Cleaning**: Unicode normalization, special character removal, whitespace cleanup
- **Intelligent Chunking**: Sentence-aware splitting with configurable chunk size (default 150 words) and overlap (default 20 words)

### 2. **Embedding Model**
- **Model**: `intfloat/multilingual-e5-small`
- **Dimensions**: 384-dimensional embeddings
- **Pooling**: Mean pooling with attention masking
- **Normalization**: L2 normalization for cosine similarity

### 3. **Vector Database**
- **Platform**: Pinecone
- **Indexing**: Serverless index with cosine similarity metric
- **Metadata**: Stores chunk text, language, tokens, and document IDs
- **Filtering**: Document-level filtering during search

### 4. **Search & Retrieval**
- **Semantic Search**: Vector similarity using cosine distance
- **BM25 Reranking**: 
  - Language-specific tokenization
  - IDF calculation for term importance
  - Score normalization (min-max with negative handling)
  - Multi-language query support
- **Hybrid Approach**: Semantic retrieval followed by keyword-based reranking

### 5. **Language Handling**
- **Language Detection**: `langdetect` library for automatic language identification
- **Multi-lingual Tokenization**: Language-aware token filtering
- **Cross-lingual Queries**: Query translation support for Hindi and Bengali
- **Language-Preserving Responses**: LLM responds in the user's original language

### 6. **LLM Integration**
- **Model**: Mistral AI (ministral-8b-latest)
- **Context Window**: Optimized for chunk-based context
- **Response Generation**: Source-cited, markdown-formatted answers
- **Fallback Handling**: Graceful degradation when LLM unavailable

## Local Setup

### Prerequisites

```bash
Python 3.10+
LibreOffice (for .doc file support)
Pinecone account (for vector database)
Mistral AI API key (optional, for LLM responses)