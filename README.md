# LangChain-DocumentGPT
LangChain 으로 챗봇 만들기

# RAG
Retrieval Augmented Generation (검색증강생성)

- RAG 는 특정 라이브러리나 프레임워크 이름이 아니라
대규모 언어 모델(LLM)이 답변을 생성하기 전, 외부 지식 베이스(데이터베이스)에서 관련 정보를 검색하여 활용하는 기술

- LLM(대형 언어 모델)의 알려진 문제점
  - 답이 없을때 허위 정보 제공
  - 사용자가 구체적이고 최신의 응답을 기대할 때 out-of-date나 일반적인 정보 제공
  - 신뢰할 수 없는 출처로부터 응답 생성
등 의 문제를 해결하기 위한 접근 방식

- RAG 를 수행하는 방법은 다양해서, 어떤 방식으로 RAG 를 구현할지는 Domain에 따라 결정해야 한다.

# RAG의 기본 구조
1. Retrieval(추출)
  - 사용자의 질문이나 컨텍스트를 입력으로 받아서, 이와 관련된 외부 데이터를 검색하는 단계
    <img width="1653" height="580" alt="image" src="https://github.com/user-attachments/assets/a936dac6-863d-4b3d-b76d-9ee33b50b2e3" />
  - RAG 의 첫번째 단계인 Retrieval 의 일반적인 과정
    1. data source 에서 데이터 load
    2. 데이터는 split 하면서 transform
    3. transform 한 데이터를 embed.
    4. embed 된 데이터를 store 에 저장.
    5. 검색(질의) 가 입력되면 store 에서 관련 문서들을 retrieve!


2. Augmented Generation
   - 검색된 데이터를 기반으로 LLM 모델이 사용자의 질문에 답변을 생성하는 단계






# Reference
https://github.com/tetrapod0/RAG_with_lm_studio
https://aws.amazon.com/what-is/retrieval-augmented-generation/
