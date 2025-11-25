import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool, Tool
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from documents import load_documents


# Create a FAISS index using L2 (Euclidean) distance with the same #dimensionality
# as the output of the embedding function (e.g., length of a single embedded #vector)
def build_vector_store(
    openai_api_key: SecretStr, embedding_model: str = "text-embedding-3-large"
) -> VectorStore:
    """
    Initialize an embedding model for reducing the dimension of documents.
    Initialize a vector store for the resulting vectors
    :param openai_api_key: OpenAI API key
    :param embedding_model: Embedding model to use
    :return:
    """
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=embedding_model)
    dimension = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(dimension)

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def build_rag_tool(
    openai_api_key: SecretStr, directory_path: str, name: str, description: str
) -> Tool:
    """
    Load all documents from the given directory and store them in a vector store.
    :param openai_api_key: OpenAI API key
    :param directory_path: the path containing the documents
    :param name: the name of the retriever tool
    :param description: the description of the retriever tool
    :return: A retriever tool for RAG
    """
    vector_store = build_vector_store(openai_api_key)
    docs = load_documents(directory_path)
    vector_store.add_documents(documents=docs)
    retriever = vector_store.as_retriever()
    return create_retriever_tool(retriever, name, description)
