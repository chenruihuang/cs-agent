from sentence_transformers import CrossEncoder

TOP_K_RERANK = 5      # 第二阶段：Reranker 取前 5 个

_model = None
def get_model():
    global _model
    if _model is None:
        _model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
    return _model

def rerank(query: str, docs: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    model = get_model()
    """第二阶段：Reranker 精排"""
    # 构造 (query, passage) 对
    pairs = [(query, doc["text"]) for doc in docs]
    # CrossEncoder 打分
    scores = model.predict(pairs)
    
    # 把分数挂回去并排序
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)
    
    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return docs[:top_k]
