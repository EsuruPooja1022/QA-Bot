import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os
from transformers import pipeline
from unstructured.partition.auto import partition
from pathlib import Path

# 1. Set up the LLM and Embedding Model
# The llm_pipeline is configured for text-to-text generation using the flan-t5-small model.
# This model is well-suited for summarization and Q&A tasks.
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)

# Initialize the Sentence Transformer model for embeddings.
# 'all-MiniLM-L6-v2' is a small, efficient model suitable for semantic search tasks.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. PDF Processing and Chunking (Improved, smarter approach)
def process_pdf(pdf_path):
    """
    Extracts text from a PDF and chunks it intelligently based on document structure.
    
    This uses the 'unstructured' library which can parse a PDF and
    identify logical document elements like paragraphs, titles, and tables.
    This is a significant improvement over the naive character-based splitting
    and is crucial for accurate retrieval.
    """
    print("Partitioning PDF into structural elements...")
    try:
        # The 'unstructured' library handles the complexity of parsing the PDF.
        elements = partition(pdf_path)
        chunks = [str(el) for el in elements if len(str(el)) > 50] # Filter out very small chunks
        print(f"PDF processed successfully. Found {len(chunks)} text chunks.")
        return chunks
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

# 3. Vector Store Creation
def create_vector_store(chunks):
    """
    Creates a FAISS vector store from text chunks.
    
    This function embeds each text chunk and builds a FAISS index for efficient
    similarity searching.
    """
    print("Creating embeddings and building FAISS index...")
    # Generate embeddings for each chunk
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    print("FAISS index created successfully.")
    return index, embeddings

# 4. Retrieval and Answer Generation (with improved prompting)
def retrieve_chunks(query, index, embeddings, chunks, k=10): # Increased k to 10
    """
    Retrieves the top k most relevant text chunks for a given query.
    
    This function first embeds the query, then uses the FAISS index to find the
    closest text chunks based on the embedding similarity.
    """
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Get the top k chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    return retrieved_chunks

def generate_answer_with_context(query, context):
    """
    Generates a full answer using a generative LLM based on the provided context.
    
    This version uses a more effective prompting strategy for the T5 model.
    """
    print("\nSending request to LLM...")
    
    # T5 models are instruction-tuned and work best with a structured prompt.
    # The prompt is now much more explicit and instructs the model to synthesize the information.
    prompt = f"Please synthesize the following context to answer the question. Provide a comprehensive answer. question: {query} context: {context}"
    
    # Use the generative pipeline to create a comprehensive answer
    generated_text = llm_pipeline(
        prompt,
        max_length=512,  # Set a reasonable max length for concise answers
        num_return_sequences=1,
        do_sample=False, # T5 models often perform better without sampling for this task
    )
    
    answer = generated_text[0]['generated_text'].strip()
    
    return answer

def main():
    # Path to the 500-page manual provided for the assignment.
    # We use pathlib for a more robust path handling.
    pdf_path = Path("draft-oasis-e1-manual-04-28-2024.pdf")
    
    # Check if the PDF file exists and provide a helpful error message.
    if not pdf_path.exists():
        print(f"Error: The PDF file '{pdf_path.name}' was not found. Please place it in the same directory as this script.")
        return

    # Process the PDF and create the vector store
    chunks = process_pdf(str(pdf_path))
    if not chunks:
        print("Could not process PDF. Exiting.")
        return
        
    index, embeddings = create_vector_store(chunks)

    print("\nQ&A bot is ready. Ask a question about the manual.")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting.")
            break
        
        # 1. Retrieve the most relevant chunks from the PDF
        retrieved_chunks = retrieve_chunks(query, index, embeddings, chunks)
        
        # 2. Combine the retrieved chunks into a single context
        context = " ".join(retrieved_chunks)
        
        # 3. Generate the answer using the combined context and the query
        answer = generate_answer_with_context(query, context)
        
        print("\n--- Answer ---")
        print(answer)
        print("--- End of Answer ---")

if __name__ == "__main__":
    main()
