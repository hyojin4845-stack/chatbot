# 문서 기반 RAG 챗봇

`DATA` 폴더의 문서를 검색하고, 검색된 문서 내용만 근거로 답변하는 Streamlit RAG 챗봇입니다.

## 주요 기능

- PDF와 일반 텍스트 문서 자동 로딩
- OpenAI `text-embedding-3-small` 임베딩
- LangChain `InMemoryVectorStore` 기반 검색
- OpenAI `gpt-4o-mini` 답변 생성
- 벡터 검색과 키워드 검색 결합
- 답변에 출처 파일명, 페이지, 근거 문장 표시
- 문서에 없는 내용은 추측하지 않도록 구성

## 로컬 실행

Python 3.11과 `uv`를 사용합니다.

1. `.env` 파일에 OpenAI API 키를 입력합니다.

```env
OPENAI_API_KEY=여기에_OpenAI_API_키_입력
```

2. Streamlit 앱을 실행합니다.

```powershell
uv run --no-project streamlit run app.py
```

## Streamlit Cloud 배포

Streamlit Cloud의 앱 설정에서 `Advanced settings` → `Secrets`에 다음 내용을 입력합니다.

```toml
OPENAI_API_KEY = "실제_OpenAI_API_키"
```

API 키는 `README.md`, 소스 코드, GitHub 저장소에 직접 입력하지 않습니다.

## 프로젝트 구조

```text
chatbot/
├─ DATA/                         # 검색할 문서
├─ app.py                        # Streamlit 앱
├─ pyproject.toml                # 프로젝트 및 의존성 설정
├─ uv.lock                       # 잠금된 의존성 목록
└─ .streamlit/
   └─ secrets.toml.example       # Secrets 입력 예시
```

## 사용 모델

- 임베딩: `text-embedding-3-small`
- 답변: `gpt-4o-mini`
- 벡터 저장소: `InMemoryVectorStore`
