# app/rag/retriever.py
from chromadb import PersistentClient            # 之前（本地文件）
from chromadb import HttpClient
# app/rag/retriever.py（先 PersistentClient 版）
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.core.config import settings

COLLECTION_NAME = "rag_docs"
TOP_K_RETRIEVE = 20   # 第一阶段：向量检索取 20 个

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        _collection = _client.get_collection(COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-small-zh-v1.5", device="cpu"))
    return _collection

def retrieve(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    coll = get_collection()
    """第一阶段：向量粗检索"""
    results = coll.query(query_texts=[query], n_results=top_k)
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "page": meta["page"],
            "vector_score": 1 - dist,
        })
    return retrieved
