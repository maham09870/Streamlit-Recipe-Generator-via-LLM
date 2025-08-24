import os
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM

load_dotenv()

# ✅ Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

genai.configure(api_key=api_key)

# ---- Custom Classes ----

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text):
        return self._embed_text(text)

    def _embed_text(self, text):
        # Uses Google's embedding model
        response = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        return response['embedding']


class GeminiLLM(LLM):
    def _call(self, prompt, stop=None):
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self):
        return {"model": "gemini-2.5-flash"}

    @property
    def _llm_type(self):
        return "gemini"


# ---- RAG Pipeline ----
class RAGPipeline:
    def __init__(self):
        self.embeddings = GeminiEmbeddings()
        self.llm = GeminiLLM()
        self.vectorstore = None
        self.qa_chain = None

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    def create_vectorstore(self, content, is_pdf=False):
        # Extract text if PDF
        text = self.extract_text_from_pdf(content) if is_pdf else content

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in docs]

        # Build vectorstore
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)

        retriever = self.vectorstore.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)

    def query(self, question):
        if not self.qa_chain:
            return "⚠️ Please upload a document first!"
        return self.qa_chain.run(question)
