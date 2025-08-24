from rag_pipeline import RAGPipeline

# Step 1: Initialize RAG pipeline
rag = RAGPipeline()
print("✅ RAGPipeline initialized successfully!")

# Step 2: Add a sample text (simulate document upload)
sample_text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
AI systems are used in various applications such as healthcare, finance, and robotics.
One popular AI technique is Machine Learning, which allows systems to learn from data.
"""
rag.create_vectorstore(sample_text, is_pdf=False)
print("✅ Vector store created from sample text!")

# Step 3: Test a query
question = "What is AI used for?"
answer = rag.query(question)
print(f"❓ Question: {question}")
print(f"✅ Answer: {answer}")
