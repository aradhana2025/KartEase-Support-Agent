from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "kartease_policies"


# Load Markdown documents
documents = []

for file_path in sorted(DATA_DIR.glob("*.md")):
    text = file_path.read_text(encoding="utf-8")

    documents.append(
        Document(
            page_content=text,
            metadata={"source": file_path.name}
        )
    )

if not documents:
    raise ValueError("No Markdown policy documents found in data/")

print(f"Loaded {len(documents)} policy documents.")


# Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks.")


# Create the embeddings configuration
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# Open the persistent ChromaDB collection
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(DB_DIR)
)


# Check whether vectors already exist
existing_count = vector_store._collection.count()

if existing_count > 0:
    print(
        f"Knowledge base already contains {existing_count} chunks."
    )
    print("Reusing the existing knowledge base.")
else:
    print("No existing vectors found. Creating the knowledge base...")

    vector_store.add_documents(chunks)

    print(f"Added {len(chunks)} chunks to ChromaDB.")

print("Knowledge base ready!")
print(f"Database location: {DB_DIR}")