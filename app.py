"""DATA 폴더의 문서를 검색해 답변하는 간단한 RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
from pathlib import Path

# 사용자 프로필에 쓰기 권한이 없어도 실행되도록 Streamlit 설정을 프로젝트 안에 둡니다.
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("STREAMLIT_CONFIG_DIR", str(BASE_DIR / ".streamlit"))
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from pypdf import PdfReader


# 프로젝트 최상위 폴더와 문서 폴더를 기준으로 경로를 만듭니다.
DATA_DIR = BASE_DIR / "DATA"


def get_openai_api_key() -> str | None:
    """배포 환경에서는 Streamlit Secrets, 로컬에서는 .env에서 키를 읽습니다."""
    try:
        cloud_key = st.secrets.get("OPENAI_API_KEY")
    except StreamlitSecretNotFoundError:
        cloud_key = None

    return cloud_key or os.getenv("OPENAI_API_KEY")


def read_text_file(path: Path) -> str:
    """텍스트 파일을 UTF-8로 읽고, 실패하면 한국어 Windows 인코딩을 사용합니다."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp949")


def load_documents() -> list[Document]:
    """DATA 폴더 안의 PDF와 일반 텍스트 문서를 LangChain 문서로 읽습니다."""
    documents: list[Document] = []

    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue

        relative_name = str(path.relative_to(DATA_DIR))
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": relative_name,
                                "page": page_number,
                            },
                        )
                    )
        elif suffix in {".txt", ".md", ".csv", ".json", ".log"}:
            text = read_text_file(path).strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": relative_name, "page": None},
                    )
                )
        else:
            # 이미지나 기타 바이너리 파일은 현재 텍스트 RAG 대상이 아니므로 건너뜁니다.
            st.warning(f"지원하지 않는 파일 형식이라 건너뜁니다: {relative_name}")

    return documents


def format_source(metadata: dict) -> str:
    """문서 메타데이터를 화면에 표시할 출처 문자열로 바꿉니다."""
    source = metadata.get("source", "알 수 없는 파일")
    page = metadata.get("page")
    return f"{source} (p. {page})" if page else source


def evidence_sentence(
    text: str,
    question: str = "",
    max_length: int = 240,
) -> str:
    """검색된 청크에서 질문과 가장 관련 있는 근거 문장을 뽑습니다."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
    question_terms = {
        re.sub(r"\s+", "", term)
        for term in re.findall(r"[가-힣A-Za-z0-9]{2,}", question)
        if term not in {"무엇인가요", "무엇", "인가요", "시", "은", "는", "이", "가"}
    }

    def relevance(sentence: str) -> int:
        normalized_sentence = re.sub(r"\s+", "", sentence)
        return sum(term in normalized_sentence for term in question_terms)

    candidates = [sentence.strip() for sentence in sentences if sentence.strip()]
    evidence = max(candidates, key=relevance, default=cleaned)
    return evidence[:max_length] + ("..." if len(evidence) > max_length else "")


@st.cache_resource(show_spinner=False)
def build_rag() -> tuple[InMemoryVectorStore, list[Document], int, int]:
    """문서를 읽고 임베딩한 뒤, InMemoryVectorStore를 만듭니다."""
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    raw_documents = load_documents()
    if not raw_documents:
        raise ValueError("DATA 폴더에서 읽을 수 있는 문서를 찾지 못했습니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "。", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)

    # 지정한 임베딩 모델로 청크를 벡터로 바꿉니다.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        # 로컬 토큰화기를 사용하지 않고 원문을 OpenAI API에 전달합니다.
        # 따라서 tiktoken 인코딩 파일이나 transformers 모델을 별도로 다운로드하지 않습니다.
        check_embedding_ctx_length=False,
    )
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    return vector_store, chunks, len(raw_documents), len(chunks)


def retrieve_documents(
    vector_store: InMemoryVectorStore,
    chunks: list[Document],
    question: str,
) -> list[Document]:
    """벡터 검색과 키워드 검색을 합쳐 제목보다 본문 근거를 우선합니다."""
    # 의미가 비슷한 문서를 찾는 기본 검색입니다.
    vector_documents = vector_store.similarity_search(question, k=8)

    # PDF 표·목차는 임베딩 검색에서 제목 청크가 과하게 선택될 수 있어
    # 질문의 핵심 단어가 실제로 포함된 청크도 함께 찾습니다.
    query_terms = [
        term
        for term in re.findall(r"[가-힣A-Za-z0-9]{2,}", question)
        if term not in {"무엇인가요", "무엇", "인가요", "시", "은", "는", "이", "가"}
    ]
    normalized_terms = {re.sub(r"\s+", "", term) for term in query_terms}

    def keyword_score(document: Document) -> int:
        normalized_text = re.sub(r"\s+", "", document.page_content)
        score = 0
        for term in normalized_terms:
            if term in normalized_text:
                score += 2
            elif len(term) >= 3 and term[:2] in normalized_text:
                score += 1
        return score

    keyword_documents = [
        document
        for document in sorted(chunks, key=keyword_score, reverse=True)
        if keyword_score(document) > 0
    ][:6]

    retrieved_documents: list[Document] = []
    seen: set[tuple[str, int | None, str]] = set()
    for document in [*keyword_documents, *vector_documents]:
        key = (
            str(document.metadata.get("source")),
            document.metadata.get("page"),
            document.page_content,
        )
        if key not in seen:
            seen.add(key)
            retrieved_documents.append(document)
        if len(retrieved_documents) >= 8:
            break

    return retrieved_documents


def answer_question(
    vector_store: InMemoryVectorStore,
    chunks: list[Document],
    question: str,
) -> tuple[str, list[Document]]:
    """검색 결과만 근거로 최신 LangChain Runnable 방식으로 답변합니다."""
    retrieved_documents = retrieve_documents(vector_store, chunks, question)
    context = "\n\n".join(
        f"[출처: {format_source(document.metadata)}]\n{document.page_content}"
        for document in retrieved_documents
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 문서 기반 질문 답변 도우미입니다.
아래 CONTEXT에 있는 내용만 사용해 한국어로 답변하세요.
CONTEXT에 답이 없거나 불확실하면 반드시 '문서에서 확인할 수 없습니다.'라고 답하세요.
상식, 추측, 외부 지식을 보태지 마세요.

CONTEXT:
{context}""",
            ),
            ("human", "질문: {question}"),
        ]
    )
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=get_openai_api_key(),
    )
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, retrieved_documents


def main() -> None:
    """Streamlit 화면을 구성합니다."""
    load_dotenv(BASE_DIR / ".env")
    st.set_page_config(page_title="문서 RAG 챗봇", page_icon="📚")
    st.title("📚 문서 기반 RAG 챗봇")
    st.caption("DATA 폴더의 문서만 근거로 답변합니다.")

    if not get_openai_api_key():
        st.error(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "로컬에서는 .env에, Streamlit Cloud에서는 App settings > Secrets에 입력하세요."
        )
        st.stop()

    try:
        with st.spinner("문서를 읽고 검색 인덱스를 준비하는 중입니다..."):
            vector_store, chunks, document_count, chunk_count = build_rag()
        st.success(f"문서 {document_count}개, 검색 청크 {chunk_count}개를 준비했습니다.")
    except Exception as exc:  # 사용자가 원인을 확인할 수 있도록 화면에 표시합니다.
        st.error(f"문서 준비 중 오류가 발생했습니다: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("**출처 및 근거**")
                for source, evidence in message["sources"]:
                    st.caption(f"📄 {source}\n\n> {evidence}")

    question = st.chat_input("문서에 대해 질문해 보세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 근거를 찾는 중입니다..."):
            try:
                answer, documents = answer_question(vector_store, chunks, question)
                sources = [
                    (
                        format_source(document.metadata),
                        evidence_sentence(document.page_content, question),
                    )
                    for document in documents
                ]
                st.markdown(answer)
                st.markdown("**출처 및 근거**")
                for source, evidence in sources:
                    st.caption(f"📄 {source}\n\n> {evidence}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"답변 생성 중 오류가 발생했습니다: {exc}")


if __name__ == "__main__":
    main()
