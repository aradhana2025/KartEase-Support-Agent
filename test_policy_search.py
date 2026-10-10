
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

DB_DIR = Path(__file__).parent / "chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = Chroma(
    collection_name="kartease_policies",
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)


def test_knowledge_base_exists():
    assert DB_DIR.exists()


def test_return_policy_retrieval():
    results = vector_store.similarity_search(
        "What is the return policy for electronics?",
        k=3,
    )

    assert len(results) > 0

    combined_text = " ".join(
        document.page_content.lower()
        for document in results
    )

    assert "electronics" in combined_text
    assert any(
        "returns_and_refunds.md" in str(document.metadata)
        for document in results
    )


def test_refund_policy_retrieval():
    results = vector_store.similarity_search(
        "How are refunds processed?",
        k=3,
    )

    assert len(results) > 0

    combined_text = " ".join(
        document.page_content.lower()
        for document in results
    )

    assert "refund" in combined_text


def test_warranty_policy_retrieval():
    results = vector_store.similarity_search(
        "What is the manufacturer warranty for electronics?",
        k=3,
    )

    assert len(results) > 0

    combined_text = " ".join(
        document.page_content.lower()
        for document in results
    )

    assert "warranty" in combined_text