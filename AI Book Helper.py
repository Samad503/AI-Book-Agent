import streamlit as st
import streamlit.components.v1 as components
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
import os

# ---- Title ----
st.title("📚 AI Agent for Your Book")

# ---- API Key ----
os.environ["ZHIPUAI_API_KEY"] = "c49c2ec869db4fe1ad1c2671b95d4bb1.rqM6wykOXfUAp6TA"

# Helper: clean text
def clean_text(text: str) -> str:
    # Remove invalid surrogate characters safely
    safe = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return safe.replace("\n", " ").replace("\r", " ").strip()

# -------------------- Build DB --------------------
def build_book_db(pdf_file):
    """Loads the uploaded PDF, splits it into chunks, and builds a FAISS DB."""

    # Save uploaded file temporarily
    temp_path = os.path.join("./", pdf_file.name)
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Clean texts
    texts = [clean_text(doc.page_content) for doc in chunks if doc.page_content.strip()]

    # Create embeddings
    embeddings = ZhipuAIEmbeddings(api_key=os.environ["ZHIPUAI_API_KEY"])

    # Use FAISS instead of Chroma
    db = FAISS.from_texts(texts, embeddings)

    return db.as_retriever()

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# -------------------- Initialize Agent --------------------
if uploaded_file and "retriever" not in st.session_state:
    st.session_state.retriever = build_book_db(uploaded_file)
    st.success("✅ PDF processed and knowledge base built!")

if "retriever" in st.session_state:
    retriever = st.session_state.retriever

    @tool
    def load_books(query: str) -> str:
        """Answer questions by retrieving relevant text from the uploaded book."""
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs[:5]])

    @tool
    def quiz_from_books(topic: str = "machine learning") -> str:
        """Generate 5 quiz questions from the uploaded book content related to a topic."""
        docs = retriever.get_relevant_documents(topic)
        text = "\n\n".join([doc.page_content for doc in docs[:5]])
        return f"Make 5 quiz questions from this:\n{text}"

    search_tool = DuckDuckGoSearchRun()
    chat = ChatZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"], model_name="GLM-4.5")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=chat,
        tools=[search_tool, load_books, quiz_from_books],
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool, load_books, quiz_from_books],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # ---- Query Box ----
    query = st.text_input("💬 Ask me something about your PDF:")

    # Only run query when new input is given
    if query and st.session_state.get("last_query") != query:
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": query})
            st.session_state.answer = response["output"]
            st.session_state.last_query = query

    # Display answer if available
    if "answer" in st.session_state:
        st.markdown("### 🤖 Answer:")
        st.write(st.session_state.answer)

        # ---- Speak / Stop Buttons ----
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔊 Speak Answer"):
                js_code = f"""
                <script>
                    var utterance = new SpeechSynthesisUtterance({st.session_state.answer!r});
                    utterance.lang = "en-US";
                    speechSynthesis.speak(utterance);
                </script>
                """
                components.html(js_code, height=0, width=0)

        with col2:
            if st.button("🛑 Stop Speaking"):
                js_code = """
                <script>
                    speechSynthesis.cancel();
                </script>
                """
                components.html(js_code, height=0, width=0)

