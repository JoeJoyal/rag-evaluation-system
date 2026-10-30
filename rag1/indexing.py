from dotenv import load_dotenv
from file_reader import extract_paragraph
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

chunks = extract_paragraph("pdfs\Joe_Wilson_Resume_v2.pdf")
vector_store.add_texts(texts=chunks)
print(f"indexed {len(chunks)} chunks")