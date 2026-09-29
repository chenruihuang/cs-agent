import json, time
from pathlib import Path
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.generator import rewrite_query
from app.core.config import settings

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "eval_output"     # 写报告

def evaluate_retrieval(questions_path: str = "eval/questions.json"):
    """评估检索质量：答案是否在 Top-K 中（Hit Rate）"""
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    hits5 = hits20 = 0
    mrr_sum = 0.0
    rows = []

    for q in questions:
        rewritten_question = rewrite_query(q["question"])
        docs = retrieve(rewritten_question, top_k=settings.top_k_retrieve)
        top5 = rerank(rewritten_question, docs, top_k=settings.top_k_rerank)
        pages = [d["page"] for d in docs[:20]]
        target_pages = q["answer_page"]
        if isinstance(target_pages, int):            # 兼容单页是 int 的情况
            target_pages = [target_pages]

        # 命中 = 检索页包含任一答案页，取最早命中位置
        rank = min([pages.index(p) + 1 for p in target_pages if p in pages] or [21])
        hits5 += rank <= 5
        hits20 += rank <= 20
        mrr_sum += 1 / rank if rank <= 20 else 0
        rows.append({"id": q["id"], "rank": rank, "hit5": rank <= 5})
        print(f"{q['id']}: 结果页：[{pages}] 答案页[{target_pages}] rank={rank} {'✅' if rank <= 5 else '❌'}")

    n = len(questions)
    report = {
        "total": n, "hit5": hits5, "hit20": hits20,
        "hit5_rate": round(hits5 / n, 4), "hit20_rate": round(hits20 / n, 4),
        "mrr": round(mrr_sum / n, 4), "rows": rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = OUTPUT_DIR / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)   # ← 没有目录就建
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"检索评估（共 {n} 题）：")
    print(f"  Hit@5  = {hits5}/{n} = {hits5/n:.1%}")
    print(f"  Hit@10 = {hits20}/{n} = {hits20/n:.1%}")
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
    evaluate_retrieval(settings.eval_path)
    # evaluate_answers()  # 需要人工逐题看
