from app.rag.reranker import rerank
from app.rag.retriever import retrieve
from app.rag.generator import generate, rewrite_query

def test_generator(query: str, verbose: bool = True) -> str:
    
    rewritten = rewrite_query(query)
    print(f'重写结果：{rewritten}')
    retrieve_result = retrieve(rewritten)
    reranker_result = rerank(rewritten, retrieve_result)
    answer = generate(rewritten, reranker_result)
    if verbose:
        print(f"\n[回答]\n{answer}")
    return 

if __name__ == "__main__":
    print("测试开始")
    # 多轮问答
    while True:
        query = input("\n请输入问题（输入 q 退出）：")
        if query.lower() in ("q", "quit", "exit"):
            break
        test_generator(query)
    