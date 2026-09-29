import json
import chromadb
import os
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path

# ============ 配置 ============
CHUNKS_PATH = "chunks/chunks.json"
DB_PATH = "db"
COLLECTION_NAME = "rag_docs"
TOP_K = 20  # 第一阶段检索数量（给 Reranker 用，比最终需要的多）
_client = None
_collection = None
# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def get_collection():
    global _client, _collection
    if _collection is None:
        # _client = chromadb.PersistentClient(path=DB_PATH)
        _client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        _collection = _client.get_collection(COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-small-zh-v1.5", device="cpu"))
    return _collection

def get_embedding_function():
    """使用 BGE 中文 embedding 模型（本地运行）"""
    return SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
    )

    
def build_vector_db():
    global _client, _collection
    # 1. 加载切分结果
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"加载 {len(chunks)} 个文本块")

    # 2. 初始化客户端（二选一，看运行环境）
    # 容器里跑（compose build-db 连独立服务）：
    _client = chromadb.HttpClient(host="chromadb", port=8000)
    # 本地调试跑：
    # _client = chromadb.PersistentClient(path=DB_PATH)

    # 3. 删除旧集合（在 client 上操作，不是 collection！）
    try:
        _client.delete_collection(COLLECTION_NAME)
        print("已删除旧集合，重建中...")
    except Exception as e:
        print(f"（无旧集合或删除失败：{e}）")

    # 4. 创建集合（也在 client 上）
    collection = _client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    # 5. 批量向量化并入库（原样保留）
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"]} for c in batch],
        )
        print(f"已入库 {min(i + batch_size, len(chunks))}/{len(chunks)} 块")

    print(f"向量库构建完成，共 {collection.count()} 条记录")
    return collection

def search(query: str, collection, top_k: int = TOP_K) -> list[dict]:
    """检索相关文本块"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    
    # 整理结果
    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "page": meta["page"],
            "score": 1 - dist,  # cosine distance → similarity
        })
    return retrieved

if __name__ == "__main__":
    collection = build_vector_db()
    
    # 测试检索
    print("\n===== 检索测试 =====")
    test_query = "什么情况下，担保物权消灭"  # ← 改成你 PDF 里的真实问题
    results = search(test_query, collection, top_k=5)
    print("\n===== 结果 =====")
    print(f"\n {results} ")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] 第{r['page']}页 | 相似度 {r['score']:.4f}")
        print(r["text"][:150] + "...")