# LangChain, LLM, LangGraph 를 사용한 DocumentGPT + Memory(MemorySaver) 구현

import os
import time
import hashlib
from dotenv import load_dotenv

load_dotenv()

print(f"✅ {os.path.basename(__file__)} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\tOPENAI_API_KEY={os.getenv('OPENAI_API_KEY')[:20]}...")
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.runnables.base import RunnableLambda
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

from langchain_classic.embeddings import CacheBackedEmbeddings

from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler

# 메모리를 위해
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import BaseMessage

# ✅ LangGraph + MemorySaver
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

#──────────────────────────────────────────────────────────────────────────────
# ✅ session_state (MemorySaver)
#──────────────────────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = "useless"

if "checkpointer" not in st.session_state:
    st.session_state["checkpointer"] = MemorySaver()
checkpointer = st.session_state["checkpointer"]

# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────

class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        pass

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

prompt = ChatPromptTemplate.from_messages([
    ('system', """
            Answer the question using ONLY the following context and chat history.
            If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
    """),

    MessagesPlaceholder(variable_name='history'),

    ('human', "{question}"),
])

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────

upload_dir = r'./.cache/files'
embedding_dir = r'./.cache/embeddings'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)

def compute_file_hash(file_bytes: bytes) -> str:
    
    return hashlib.sha256(file_bytes).hexdigest()[:12]

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_bytes: bytes, file_name: str, file_hash: str):

    safe_file_name = f"{file_hash}__{file_name}"
    file_path = os.path.join(upload_dir, safe_file_name)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore(os.path.join(embedding_dir, file_hash))
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("Document GPT")

st.markdown("""
안녕하세요!
이 챗봇을 사용하여 여러분들의 파일들에 대해 물어보세요
""")

with st.sidebar:
    file = st.file_uploader(
        label="upload a .txt .pdf .docx file",
        type=['pdf', 'txt', 'docx']
    )

def send_message(message, role):
    with st.chat_message(role):
        st.markdown(message)

# ✅ LangGraph State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def paint_history_from_checkpointer(graph, config):
    """
    MemorySaver(checkpointer)에 저장된 messages를 읽어 화면에 그대로 렌더링
    """
    try:
        snapshot = graph.get_state(config)
        msgs = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    except Exception:
        msgs = []

    for m in msgs:
        if isinstance(m, HumanMessage):
            send_message(m.content, "human")
        elif isinstance(m, AIMessage):
            send_message(m.content, "ai")
        else:
            send_message(getattr(m, "content", str(m)), "assistant")

if file:
    file_bytes = file.getvalue()
    file_hash = compute_file_hash(file_bytes)

    st.session_state["thread_id"] = f"doc::{file_hash}::{file.name}"

    retriever = embed_file(file_bytes, file.name, file_hash)

    send_message("준비되었습니다. 질문해보세요", "ai")

    def run_chain(state: State):
        question = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        chain = (
            {
                "history": lambda _ : state["messages"],
                "context": retriever | RunnableLambda(format_docs),
                "question": lambda _ : question,
            }
            | prompt
            | llm
        )

        result = chain.invoke(question)

        return {
            "messages": [
                AIMessage(content=result.content),
            ]
        }

    graph_builder = StateGraph(State)
    graph_builder.add_node("run", run_chain)
    graph_builder.add_edge(START, "run")
    graph_builder.add_edge("run", END)

    graph = graph_builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    paint_history_from_checkpointer(graph, config)

    message = st.chat_input("업로드한 file 에 대해 질문을 남겨보세요...")
    if message:
        send_message(message, "human")

        with st.chat_message("ai"):
            _ = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config
            )
else:
    pass
