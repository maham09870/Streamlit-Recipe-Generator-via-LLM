from fastapi import FastAPI, UploadFile, Form
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()
rag = RAGPipeline()

@app.post("/upload/")
async def upload_document(file: UploadFile):
    """Upload a document and create vector store"""
    content = await file.read()
    text = content.decode("utf-8")
    rag.create_vectorstore(text)
    return {"status": "Document processed and vector store created"}

@app.post("/query/")
async def query(question: str = Form(...)):
    """Ask a question from uploaded docs"""
    answer = rag.query(question)
    return {"answer": answer}
