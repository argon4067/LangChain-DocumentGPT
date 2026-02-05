# LangChain-DocumentGPT
LangChain 으로 챗봇 만들기

# RAG
## Retrieval Augmented Generation (검색증강생성)

- RAG 는 특정 라이브러리나 프레임워크 이름이 아니라
대규모 언어 모델(LLM)이 답변을 생성하기 전, 외부 지식 베이스(데이터베이스)에서 관련 정보를 검색하여 활용하는 기술

- LLM(대형 언어 모델)의 알려진 문제점
  - 답이 없을때 허위 정보 제공
  - 사용자가 구체적이고 최신의 응답을 기대할 때 out-of-date나 일반적인 정보 제공
  - 신뢰할 수 없는 출처로부터 응답 생성
등 의 문제를 해결하기 위한 접근 방식

- RAG 를 수행하는 방법은 다양해서, 어떤 방식으로 RAG 를 구현할지는 Domain에 따라 결정해야 한다.

# RAG의 기본 구조
## 1. Retrieval(추출)
사용자의 질문이나 컨텍스트를 입력으로 받아서, 이와 관련된 외부 데이터를 검색하는 단계
    <img width="1653" height="580" alt="image" src="https://github.com/user-attachments/assets/a936dac6-863d-4b3d-b76d-9ee33b50b2e3" />
    
  - RAG 의 첫번째 단계인 Retrieval 의 일반적인 과정
    1. data source 에서 데이터 load
    2. 데이터는 split 하면서 transform
    3. transform 한 데이터를 embed.
    4. embed 된 데이터를 store 에 저장.
    5. 검색(질의) 가 입력되면 store 에서 관련 문서들을 retrieve!

## 2. Augmented Generation
   - 검색된 데이터를 기반으로 LLM 모델이 사용자의 질문에 답변을 생성하는 단계
  
# RAG의 기법

## 1. Stuff Document Chain
  <img width="2140" height="1164" alt="image" src="https://github.com/user-attachments/assets/f77df65d-d444-402c-84a0-9b11ddd936b1" />
  
  여러개의 문서(Document)를 하나로 합쳐서 (stuffing) 프롬프트에 넣어 LLM 에 전달하는 기법
  
  - 장점 
    - 문서의 양이 LLM 의 Context window 보다 작다면 매우 빠르고 효율적.
  
  - 단점
    - 길이 제한 문졔: AI 모델은 입력 길이에 제한이 있음. 너무 긴 문서를 처리하려고 하면 잘리거나 오류가 발생할 수 있음
    - 정보 손실 문제: 모델이 너무 많은 정보를 한꺼번에 처리하려다보니 중요한 내용을 놓칠 수 있음

## 2. Map-Reduce chain
  <img width="2272" height="702" alt="image" src="https://github.com/user-attachments/assets/b8010626-14e6-41f3-98ab-a11e73711806" />

  질문에 대한 관련 문서 각각을 처리(Map)하고, 결과를 결합해 최종 출력을 생성(Reduce)하는 방법

  - 장점
    - 긴 문서에 적합: 각각의 문서를 요약하므로 검색한 문서가 많거나 길 때 유용
    - 정보 보전: 각각의 문서를 독립적으로 처리하므로 정보 보존에 유리

 - 단점
    - 비용: 문서별로 처리하고 최종적으로 출력을 수행하는 과정에서 AI 모델을 계속 사용해야하므로 토큰 사용이 많아질 수 있어서 비용이 증가
    - 속도: 처리해야하는 과정이 많아지기 때문에 Stuff에 비해 상대적으로 느림
  

## 3. Map-Rerank chain
   <img width="1404" height="494" alt="image" src="https://github.com/user-attachments/assets/a748832e-b744-4e65-a082-6a15e4453008" />
   
   Map-rerank 흐름
   
    1. Map 단계
       각 Document 를 독립적으로 LLM 에 전달
       답변 + 관련성 점수 생성.
    
    2. Rerank 단계
        점수를 기준으로 가장 좋은 답변 선택.
   


## 4. Refine Document chain







# Reference
https://github.com/tetrapod0/RAG_with_lm_studio
https://aws.amazon.com/what-is/retrieval-augmented-generation/
https://velog.io/@pysun/LangChain-RAG-%EA%B8%B0%EB%B2%95-Stuff
