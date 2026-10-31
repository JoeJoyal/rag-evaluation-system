from dotenv import load_dotenv
from file_reader import extract_paragraph
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize BGE-M3 embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# Initialize vector size
vector_store = InMemoryVectorStore(embeddings)

# Extract paragraphs from PDF
chunks = extract_paragraph("pdfs\Joe_Wilson_Resume_v2.pdf")

# Convert text into embeddings and store vectors
vector_store.add_texts(texts=chunks)
print(f"indexed {len(chunks)} chunks")

# Test similarity search
# results = vector_store.similarity_search("What is Joe's Python experience?", k=3)

# Display retrieved results
# for result in results:
#     print(result.page_content)
