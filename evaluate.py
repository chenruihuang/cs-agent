import json
from rag_pipeline import retrieve, rerank, rag_query, rewrite_query, hybrid_search


def evaluate_retrieval(questions_path: str = "eval/questions.json"):
    """评估检索质量：答案是否在 Top-K 中（Hit Rate）"""
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    hit_at_5 = 0
    hit_at_10 = 0
    mrr = 0  # 平均倒数排名
    
    for q in questions:
        rewritten = rewrite_query(q["question"])
        docs = retrieve(rewritten, top_k=20)
        top_docs = rerank(rewritten, docs, top_k=5)

        # docs = retrieve(q["question"], top_k=20)
        # top_docs = rerank(q["question"], docs, top_k=5)
        
        # 检查答案页码是否出现在检索结果中
        answer_pages = set(q["answer_page"])
        
        for rank, doc in enumerate(top_docs, 1):
            if doc["page"] in answer_pages:
                if rank <= 5:
                    hit_at_5 += 1
                if rank <= 10:
                    hit_at_10 += 1
                mrr += 1.0 / rank
                break
    
    n = len(questions)
    print(f"检索评估（共 {n} 题）：")
    print(f"  Hit@5  = {hit_at_5}/{n} = {hit_at_5/n:.1%}")
    print(f"  Hit@10 = {hit_at_10}/{n} = {hit_at_10/n:.1%}")
    print(f"  MRR     = {mrr/n:.4f}")

def evaluate_answers(questions_path: str = "eval/questions.json"):
    """评估回答质量：人工检查要点覆盖率"""
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    print("回答质量评估（逐题人工判断）：")
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(questions)}] 问题：{q['question']} 难度：{q['difficulty']}")
        print(f"  期望要点：{q['key_points']}")
        answer = rag_query(q["question"], verbose=False)
        print(f"  系统回答：{answer}")
        # 这里可以人工打分，或用 LLM-as-judge 自动评估
        # input("  按回车继续下一题...")

if __name__ == "__main__":
    evaluate_retrieval()
    # evaluate_answers()  # 需要人工逐题看
