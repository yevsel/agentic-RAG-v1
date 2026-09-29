from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

# URLs we will be scraping
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm"
]

# WebBaseLoader(url).load() returns a LIST containing one Document per url
# (even though it's just one page), so this comprehension produces a list
# of lists: one [Document] per url.
docs_per_url = [WebBaseLoader(url).load() for url in urls]

# Flatten docs_per_url into one flat list of Document objects, one per url,
# so downstream code (the splitter) can treat all pages the same way.
flattened_docs = []
for url_docs in docs_per_url:
    for doc in url_docs:
        flattened_docs.append(doc)

# chunk_size is counted in TOKENS here (via tiktoken), not characters -
# 250 tokens per chunk, no overlap between neighboring chunks.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

# Each flattened_docs entry (one full page) gets split into many smaller chunks
chunked_docs = text_splitter.split_documents(flattened_docs)

# print(len(chunked_docs))


# Embedding - We want to embed our chunks into ChromaDB
# vector_store = Chroma.from_documents(
#     documents=chunked_docs,
#     collection_name="agentic_rag_v1",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma"
# )

# We want to get a retriever so we can make similarity searches
chroma_docs_retriever = Chroma(
    collection_name="agentic_rag_v1",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings()
).as_retriever()