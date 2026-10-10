
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

db = Chroma(
    collection_name="kartease_policies",
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db")
)

records = db.get(include=["documents", "metadatas"])

print("Total stored chunks:", len(records["ids"]))
print("Unique document chunks:", len(set(records["documents"])))

print("\nChunks by source:")
sources = {}

for metadata in records["metadatas"]:
    source = metadata.get("source", "Unknown")
    sources[source] = sources.get(source, 0) + 1

for source, count in sources.items():
    print(f"{source}: {count}")