import streamlit as st
from rag_pipeline import RAGPipeline
import tempfile

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

rag = st.session_state.rag

st.title("📚 Research Assistant (RAG + Gemini)")

uploaded_files = st.file_uploader("Upload documents (PDF or Text)", type=["txt", "md", "pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                rag.create_vectorstore(temp_path, is_pdf=True)
            else:
                text = uploaded_file.read().decode("utf-8")
                rag.create_vectorstore(text, is_pdf=False)
    st.success("✅ Documents processed successfully!")

st.subheader("💬 Chat with your documents")
question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not rag.qa_chain:
        st.error("⚠️ Please upload at least one document first!")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            answer = rag.query(question)
            st.session_state.chat_history.append({"question": question, "answer": answer})

# Show chat history
if st.session_state.chat_history:
    st.write("### 🗨️ Conversation")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Assistant:** {chat['answer']}")
        st.markdown("---")
