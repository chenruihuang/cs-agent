import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============ 配置 ============
DB_PATH = "db"
COLLECTION_NAME = "rag_docs"
TOP_K_RETRIEVE = 20   # 第一阶段：向量检索取 20 个
TOP_K_RERANK = 5      # 第二阶段：Reranker 取前 5 个
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ============ 初始化各组件 ============

# 1. 向量库
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5", device="cpu"
)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME, embedding_function=embedding_fn
)

# 2. Reranker（交叉编码器，对 query-passage 对打分）
print("加载 Reranker 模型...")
reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
print("Reranker 加载完成")

# 3. LLM
llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ============ RAG 核心流程 ============

def retrieve(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    """第一阶段：向量粗检索"""
    results = collection.query(query_texts=[query], n_results=top_k)
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

def rerank(query: str, docs: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """第二阶段：Reranker 精排"""
    # 构造 (query, passage) 对
    pairs = [(query, doc["text"]) for doc in docs]
    # CrossEncoder 打分
    scores = reranker.predict(pairs)
    
    # 把分数挂回去并排序
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)
    
    docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return docs[:top_k]

def generate_answer(query: str, context_docs: list[dict]) -> str:
    """用 LLM 基于检索到的上下文生成答案"""
    # 拼装上下文，带页码引用
    context_str = ""
    for i, doc in enumerate(context_docs, 1):
        context_str += f"【资料{i}（第{doc['page']}页）】\n{doc['text']}\n\n"
    
    system_prompt = f"""你是一个严谨的文档问答助手。
规则：
1. 只根据下方提供的资料回答问题，不要使用你自己的知识
2. 如果资料中包含与问题相关的信息，必须完整引用并展开回答——
   资料中提到的所有相关要点都要覆盖，不要只回答其中一部分
3. 回答结构：先完整陈述资料中与该问题相关的所有内容，
   然后如果资料确实没有覆盖问题的某一方面，再指出"资料未涉及这部分"
4. 只有资料中完全没有任何相关信息时，才回答"根据现有资料无法回答这个问题"
5. 回答时在关键信息后标注来源，如（第X页）
6. 回答简洁准确，不编造信息

资料：
{context_str}"""

    response = llm.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1,  # 低温度，减少幻觉
    )
    return response.choices[0].message.content

def rewrite_query(query: str) -> str:
    resp = llm.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL"),
        messages=[{"role": "user", "content": f"""把下面的问题改写成适合法律条文检索的关键词，只输出关键词用空格分隔，必须用法言法语，不要输出其他内容。

问题：{query}
示例："16岁打工能独立签合同吗" → "十六周岁 未成年人 劳动收入 主要生活来源 完全民事行为能力 民事法律行为"""}],
        temperature=0,
    )
    return resp.choices[0].message.content

def rag_query(query: str, verbose: bool = True) -> str:
    """完整 RAG 流程：检索 → 重排 → 生成"""
    if verbose:
        print(f"\n问题：{query}")
        print(f"{'='*50}")
    
    # 第一步：向量检索
    rewritten = rewrite_query(query)
    docs = retrieve(rewritten)
    if verbose:
        print(f"[向量检索] 召回 {len(docs)} 个候选块")
    
    # 第二步：Reranker 精排
    top_docs = rerank(rewritten, docs)
    if verbose:
        print(f"[Reranker] 精排后取 Top {len(top_docs)}：")
        for i, d in enumerate(top_docs, 1):
            print(f"  {i}. 第{d['page']}页 | rerank={d['rerank_score']:.4f} | {d['text'][:50]}...")
    
    # 第三步：生成答案，先重写提问
    answer = generate_answer(query, top_docs)
    if verbose:
        print(f"\n[回答]\n{answer}")
    return answer

# ============ 运行 ============
if __name__ == "__main__":
    # 多轮问答
    while True:
        query = input("\n请输入问题（输入 q 退出）：")
        if query.lower() in ("q", "quit", "exit"):
            break
        rag_query(query)
