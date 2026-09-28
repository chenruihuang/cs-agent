import json
from rag_pipeline import retrieve, rerank, rag_query, rewrite_query

with open("eval/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

for q in questions:
    if q["id"] not in [1, 9, 13, 20]:   # 4 个失败题
        continue
    rewritten = rewrite_query(q["question"])
    docs = retrieve(rewritten, top_k=20)          # 向量检索
    pages20 = {d["page"] for d in docs}
    hit20 = bool(set(q["answer_page"]) & pages20)

    top5 = rerank(rewritten, docs, top_k=5)        # Reranker
    for i, d in enumerate(top5, 1):
        if d["page"] in q["answer_page"]:
            print(f"答案在第{i}名，rerank分数 {d['rerank_score']:.3f}")
    pages5 = {d["page"] for d in top5}
    hit5 = bool(set(q["answer_page"]) & pages5)

    print(f"Q{q['id']} 答案页{q['answer_page']} | Top20命中: {'✅' if hit20 else '❌'} | Top5命中: {'✅' if hit5 else '❌'}")
