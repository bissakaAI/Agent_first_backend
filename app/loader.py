from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("openai_key")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found! Please set it in your .env file.")

print("✅ API key loaded successfully")



PDF_PATH = "../data/tax_bills_pdfs"
CHROMA_PATH = "chroma_db"
# Load environment variables


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)
vectorstore = Chroma(
    collection_name="nigeria_tax_bills",
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)


