from rag_pipeline import RAGPipeline

# Step 1: Initialize RAG pipeline
rag = RAGPipeline()
print("✅ RAG Pipeline Initialized Successfully!")

# Step 2: Provide sample text for testing
sample_text = """
Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from data,
identify patterns, and make decisions with minimal human intervention.
"""

# Step 3: Create vectorstore from text
rag.create_vectorstore(sample_text, is_pdf=False)
print("✅ Vectorstore Created Successfully!")

# Step 4: Ask a test question
question = "What is machine learning?"
answer = rag.query(question)

print("\n📌 Question:", question)
print("🤖 Answer:", answer)
