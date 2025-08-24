import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load GEMINI_API_KEY from .env

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

print("✅ GEMINI_API_KEY Loaded Successfully!")

# Step 1: Test Embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    emb = embeddings.embed_query("Hello, how are you?")
    print(f"✅ Embedding Test Passed! (Length: {len(emb)})")
except Exception as e:
    print("❌ Embedding Test Failed:", e)

# Step 2: Test LLM (Chat)
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    response = llm.invoke("Write a short sentence about AI.")
    print("✅ LLM Test Passed! Response:", response.content)
except Exception as e:
    print("❌ LLM Test Failed:", e)
