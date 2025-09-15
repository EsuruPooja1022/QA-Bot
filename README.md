**AI-Powered Q&A Bot for OASIS-E1 Manual**
**📖 Overview**

This project implements an intelligent Q&A bot designed to answer questions about the 500-page OASIS-E1 manual. The solution leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers by combining document retrieval with a language model’s generative capabilities.

The core of this solution lies in its thoughtful data preparation and retrieval approach, with a strong emphasis on a sophisticated chunking strategy and efficient search mechanism, moving beyond a generic RAG implementation.

**⚙️ Design Decisions**
**1. Document Processing and Chunking**

**Problem:** Traditional character-based chunking can split sentences and paragraphs, harming semantic integrity and retrieval accuracy.

**Solution:** We use the unstructured library to intelligently partition the PDF. It analyzes document structure (titles, paragraphs, lists, etc.), ensuring each chunk is a coherent, self-contained unit of information for accurate retrieval.

**2. Retrieval and Vector Store**

**Problem:** Searching across thousands of chunks is computationally expensive.

**Solution:**

Use Sentence-Transformers to embed each chunk into a dense vector.

Store embeddings in FAISS, enabling fast similarity search.

Retrieve top 10 most relevant chunks per user query.

**3. Answer Generation**

**Problem:** Providing raw text chunks results in fragmented, redundant answers.

**Solution:**

Retrieved chunks are passed as context to a fine-tuned language model (google/flan-t5-small).

A custom prompt guides the model to synthesize a clear, grounded, and coherent answer strictly from the manual.

**📦 Deliverables**

app.py – Complete, functional Q&A bot script.

draft-oasis-e1-manual-04-28-2024.pdf – The 500-page PDF manual used.

requirements.txt – Python dependencies.

**🚀 Instructions to Run the Bot**
**✅ Prerequisites**

Python 3.8+ installed

Sufficient memory and disk space for model + FAISS index

**Step 1: Clone the Repository**
git clone <your-repository-url>
cd <your-repository-name>

**Step 2: Install Dependencies**

It is recommended to use a virtual environment.

pip install -r requirements.txt

**Step 3: Run the Bot**

Ensure the draft-oasis-e1-manual-04-28-2024.pdf file is in the same directory as app.py.

python app.py

The bot will preprocess the PDF, build the vector index, and get ready for interaction.

**Step 4: Interact with the Bot**

Type your question and press Enter.

To exit, type: exit or quit

**📌 Assumptions**

The PDF is text-based. Scanned/image-only PDFs may fail with the unstructured library.

The environment has sufficient resources for model downloads and FAISS indexing.

This is a demo implementation of RAG concepts, not a production-ready or UI-polished system.
