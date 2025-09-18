from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
import pyttsx3
import os
import shutil

# --- Set your API key ---
os.environ["ZHIPUAI_API_KEY"] = "f3bdae3c76994d8aa4c24586ee31acfc.Mi94xaa9CpJYQIQC"


def build_book_db():
    """
    Loads PDFs from folder, splits into chunks, creates embeddings, and stores them
    safely in Chroma DB using batches to avoid API limits and corrupted HNSW indices.
    """
 
    shutil.rmtree("./book_db", ignore_errors=True)

   
    loader = DirectoryLoader(
        path=r"C:\Users\Samad\Downloads\Books",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()

   
    for doc in docs:
        doc.page_content = doc.page_content.encode("utf-8", errors="ignore").decode("utf-8")

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = ZhipuAIEmbeddings(api_key=os.environ["ZHIPUAI_API_KEY"])
    db = Chroma(persist_directory="./book_db", embedding_function=embeddings)

  
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [doc.page_content for doc in batch]
        db.add_texts(texts)

    db.persist()
    return db.as_retriever()

retriever = build_book_db()
print("Book database built successfully!")


@tool
def load_books(query: str) -> str:
    """
    Answer questions by retrieving relevant text from the book database.
    """
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs[:5]])

@tool
def quiz_from_books(topic: str = "machine learning") -> str:
    """
    Generate 5 quiz questions from book content related to a topic.
    """
    docs = retriever.get_relevant_documents(topic)
    text = "\n\n".join([doc.page_content for doc in docs[:5]])
    return f"Make 5 quiz questions from this:\n{text}"


search_tool = DuckDuckGoSearchRun()


chat = ChatZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    model_name="GLM-4.5"
)

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


def ask_ai(query: str):
    """
    Sends a query to the AI agent, prints response, and speaks it using pyttsx3.
    """
    response = agent_executor.invoke({"input": query})
    output = response["output"]
    print("AI:", output)

    import pyttsx3

    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Get and set the desired Siri voice
    voices = engine.getProperty('voices')
    for voice in voices:
        if "Siri" in voice.name:
            engine.setProperty('voice', voice.id)
            break

    # Say your text
    engine.say(output)
    engine.runAndWait()


if __name__ == "__main__":
    Query=input("Please enter your query: ")
    ask_ai(Query)
    ask_ai(Query)
