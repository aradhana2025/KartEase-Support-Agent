import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Read the API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "Google API key not found. Check your .env file."
    )

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    google_api_key=api_key,
    temperature=0
)

# Send a test question
response = llm.invoke("Explain RAG in one sentence.")

print("Gemini connection successful!")

if isinstance(response.content, str):
    print(response.content)
else:
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block["text"])