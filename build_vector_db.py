import json
import chromadb
import os
from chromadb.utils import embedding_functions
from pathlib import Path

# ============ 配置 ============
CHUNKS_PATH = "chunks/chunks.json"
DB_PATH = "db"
COLLECTION_NAME = "rag_docs"
TOP_K = 20  # 第一阶段检索数量（给 Reranker 用，比最终需要的多）

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def get_embedding_function():
    """使用 BGE 中文 embedding 模型（本地运行）"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",  # 有 GPU 改成 "cuda"
    )

def build_vector_db():
    # 1. 加载切分结果
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"加载 {len(chunks)} 个文本块")

    # 2. 初始化 ChromaDB（持久化到本地）
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 如果集合已存在，先删除（重建）
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},  # 余弦相似度
    )

    # 3. 批量向量化并入库（分批处理，避免内存爆）
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