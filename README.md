# 🧠 RAG_APP – Retrieval-Augmented Generation Experiments

This repository showcases different approaches to **Retrieval-Augmented Generation (RAG)** and includes a simple **Flask application** for serving the models.  
The goal of this project is to explore various vector search methods, embedding strategies, and model integrations to enhance question-answering systems.

---

## 📂 Repository Structure & Descriptions

### 1. `flask_rag_app/`
A Flask-based web application that provides an interface to interact with the RAG pipelines.  
It is designed to take user queries, retrieve relevant context from stored documents, and generate responses using the selected backend model.

---

### 2. `RAG_with_CS.ipynb`
Implementation of **RAG with Cosine Similarity**.  
- Uses document embeddings.  
- Retrieves the most relevant context based on cosine similarity scores.  
- Demonstrates a lightweight retrieval mechanism without specialized vector databases.  

---

### 3. `RAG_with_Ollama_FAISS.ipynb`
Implementation of **RAG with Ollama + FAISS**.  
- Uses **Ollama** for embeddings and language generation.  
- Employs **FAISS** (Facebook AI Similarity Search) for efficient vector similarity search.  
- Optimized for larger datasets with fast nearest-neighbor retrieval.  

---

### 4. `RAG_with_HuggingFace_Ollama_FAISS.ipynb`
Implementation of **RAG with HuggingFace + Ollama + FAISS**.  
- Combines **HuggingFace Transformers** for embedding generation.  
- Uses **Ollama** as the language model for response generation.  
- Relies on **FAISS** for scalable vector similarity search.  
- Provides a hybrid setup leveraging both HuggingFace and Ollama capabilities.  

---

### 5. `.gitignore`
Specifies files and directories to be excluded from version control, such as temporary files, environment settings, and cache data.

---

## 🎯 Key Highlights

- Multiple RAG implementations with different similarity and embedding backends.  
- Comparison between **cosine similarity**, **FAISS-based retrieval**, and **hybrid approaches**.  
- Integration of **Flask** to provide a lightweight, interactive way to test RAG pipelines.  
- Modular structure for experimenting with different retrieval and generation strategies.  

---

## 📜 Purpose

This repository is intended for research and experimentation in the field of **RAG (Retrieval-Augmented Generation)**.  
It provides reference implementations for developers and researchers interested in building context-aware question-answering systems using modern vector search libraries and language models.  
