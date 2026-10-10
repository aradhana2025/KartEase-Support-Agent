
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

# Connect to the existing vector database
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = Chroma(
    collection_name="kartease_policies",
    persist_directory=str(DB_DIR),
    embedding_function=embeddings
)


def search_policies(question: str, k: int = 3):
    """Find relevant policy sections for a customer question."""

    results = vector_store.similarity_search(question, k=k)

    if not results:
        print("No relevant policy information found.")
        return

    print(f"\nQuestion: {question}")
    print("\nRelevant policy information:\n")

    for index, document in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown source")

        print(f"--- Result {index} ---")
        print(f"Source: {source}")
        print(document.page_content)
        print()


if __name__ == "__main__":
    question = input("Enter a customer policy question: ").strip()

    if question:
        search_policies(question)
    else:
        print("Please enter a question.")