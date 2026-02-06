# LangChain-DocumentGPT
LangChain, LangGraph 으로 문서 챗봇 만들기

----

# LangChain
<img width="565" height="222" alt="image" src="https://github.com/user-attachments/assets/cb08b293-9ab2-4cc5-a061-a2ff495df47b" />

  - 대규모 언어모델(LLM)을 더 효율적이고 유연하게 활용할 수 있도록 설계된 프레임워크 입니다.
  - 특히 chain이라는 개념을 중심으로 작동하여 LLM을 호출하는 과정을 여러 단계로 구성하여 복잡한 작업을 수행할 수 있습니다.
  - 이를 통해 대화형 AI, 검색 시스템, 데이터 분석, 작업 자동화 등 다양한 어플리케이션을 쉽게 개발할 수 있습니다. 

## LangChain의 주요 목적
  1. LLM과 외부 데이터 통합: 외부 데이터를 언어모델과 연결하여 더 스마트한 작업을 수행할 수 있습니다.

  2. 멀티스텝 작업: 여러 단계를 체인처럼 연결하여 복잡한 로직을 처리할 수 있습니다.

  3. 메모리 기능: 대화형 애플리케이션에서 문맥을 유지하도록 설계된 메모리 기능을 제공합니다.

  4. 모듈화: 다양한 모듈을 제공하여 필요에 따라 조합하거나 확장 가능하도록 설계되었습니다.

# RAG
Retrieval-Augmented Generation
 - 자연어 처리 및 생성에서 지식 기반을 활용하여 LLM이 더 정확하고 맥락에 맞는 응답을 생성하는 기법입니다.

## RAG 의 2단계

1. Retrieval(검색 증강)
  - 먼저 질문이나 요청에 관련된 정보를 외부 지식 기반에서 검색합니다.
  - 이를 통해 **사전 학습**된 언어 모델이 가지고 있는 지식에 의존하지 않고도 최신 정보나 특정 도메인의 데이터를 활용할 수 있습니다.
  
2. Augmented Generation(생성 증강)
   
- 검색된 정보를 언어 모델을 통해 응답 생성에 활용합니다.
- 이 과정에서 검색된 정보가 언어 모델에 추가적인 context를 제공하여, 더 신뢰할 수 있고 구체적인 답변을 생성 할 수 있게 합니다.

## RAG 의 장점
 1. 정확성 향상: 모델이 기억 기반이 아니라 검색된 최신 정보를 사용하여 답함으로써, 더 신뢰할 수 있는 결과를 제공합니다.
 2. 도메인 특화: 특정 도메인에 최적화된 데이터를 검색하고 이를 활용할 수 있습니다.
 3. 대규모 데이터 활용: 사전에 학습된 모델의 제한된 지식이 아니라 대규모 외부 데이터를 동적으로 활용할 수 있습니다.

## langChain 에서 LAG 구현

1. 문서 처리 및 임베딩
    - 문서를 벡터화하여 벡터 데이터베이스에 저장합니다.
     
2. 질의와 관련된 문서 검색
    - 사용자의 질문을 벡터화한 후, 벡터 데이터베이스에서 관련 문서를 검색합니다.
   
3. LLM을 통한 응답 생성
    - 검색된 문서를 context로 추가하여 GPT와 같은 언어 모델이 자연어 응답을 생성합니다.
  
----

# LLM 연결 코드
```python
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)
```

---

# 임베딩
```python
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
```

---
# DocumentGPT 구성
  - 웹 API : Streamlit
  - LLM : ChatOpenAI() - gpt-4o 모델
  - Embedding Model : OpenAIEmbeddings()
  - PDT, TXT, DOCX만 파일 업로드 지원
  - 문서 청크사이즈 : 600, overlap : 100
  - 벡터저장소 : FAISS

---

# Review
초기화면
<img width="2560" height="1528" alt="image" src="https://github.com/user-attachments/assets/c790c4ab-5630-4137-94de-91d1ebdc054f" />

파일 삽입 후 임베딩 로딩
<img width="2560" height="1528" alt="image" src="https://github.com/user-attachments/assets/4f5e93a5-01b5-490a-8144-df18b8ebafd3" />

임베딩 완료 후
<img width="2560" height="1528" alt="image" src="https://github.com/user-attachments/assets/36631c74-f2e4-4799-be5b-63f774adbc36" />
이때 @st.cache_resource(show_spinner="Embedding file...") 를 사용하여
같은 파일의 경우 다시 임베딩 하지 않기 위해 cache 에 저장한다.

질문
<img width="2560" height="1528" alt="image" src="https://github.com/user-attachments/assets/a1792c2c-c7b0-4ae4-9d1d-c8ddda0ece90" />

```python
prompt = ChatPromptTemplate.from_messages([
    ('system', """
            Answer the question using ONLY the following context and chat history.
            If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
    """),

    MessagesPlaceholder(variable_name='history'),

    ('human', "{question}"),
])
```
prompt 를 활용하여
 - context의 내용이 있는 경우 : context를 바탕으로
 - context의 내용이 없는 경우 : chat_histoy를 바탕으로
 - 위 상황에 둘 다 해당하지 않는 경우 : 추론하지 않고 모른다 라고 답한다.
 - 평문의 경우 : 평범한 챗봇

이러한 로직으로 작동합니다.




---
# Reference
https://github.com/tetrapod0/RAG_with_lm_studio
