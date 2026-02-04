# LangGraph 를 사용한 DocumentGPT + Memory(MemorySaver) 구현

import os
import time
from dotenv import load_dotenv

load_dotenv()

print(f"✅ {os.path.basename(__file__)} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\tOPENAI_API_KEY={os.getenv('OPENAI_API_KEY')[:20]}...")
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
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
# ✅ session_state
#──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
        save_message(self.message, 'ai')

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

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):

    file_content = file.read()
    file_path = os.path.join(upload_dir, file.name)

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore(os.path.join(embedding_dir, file.name))
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

# message 저장 함수
def save_message(message, role):
    st.session_state['messages'].append({'message': message, 'role': role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state['messages']:
        send_message(
            message['message'],
            message['role'],
            save=False,
        )

# ✅ LangGraph State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str

if file:
    st.session_state["thread_id"] = f"doc::{file.name}"
    retriever = embed_file(file)

    send_message("준비되었습니다. 질문해보세요", "ai", save=False)
    paint_history()

    message = st.chat_input("업로드한 file 에 대해 질문을 남겨보세요...")
    if message:
        send_message(message, 'human')

        def run_chain(state: State):
            chain = (
                {
                    "history": lambda _ : state["messages"],
                    "context": retriever | RunnableLambda(format_docs),
                    "question": lambda _ : state["question"],
                }
                | prompt
                | llm
            )

            result = chain.invoke(state["question"])

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

        with st.chat_message('ai'):
            result_state = graph.invoke(
                {
                    "question": message,
                    "messages": [HumanMessage(content=message)],
                },
                config=config
            )
else:
    st.session_state["messages"] = []
