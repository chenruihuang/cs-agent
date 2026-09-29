from openai import OpenAI,AsyncOpenAI
from app.core.config import settings

_client = OpenAI(api_key=settings.deepseek_api_key.get_secret_value(),
                 base_url=settings.deepseek_base_url)

_async_client = AsyncOpenAI(api_key=settings.deepseek_api_key.get_secret_value(),
                            base_url=settings.deepseek_base_url)

SYSTEM_PROMPT = """你是一个严谨的文档问答助手。
规则：
1. 只根据下方提供的资料回答问题，不要使用你自己的知识
2. 如果资料中包含与问题相关的信息，必须完整引用并展开回答——
   资料中提到的所有相关要点都要覆盖，不要只回答其中一部分
3. 回答结构：先完整陈述资料中与该问题相关的所有内容，
   然后如果资料确实没有覆盖问题的某一方面，再指出"资料未涉及这部分"
4. 只有资料中完全没有任何相关信息时，才回答"根据现有资料无法回答这个问题"
5. 回答时在关键信息后标注来源，如（第X页）
6. 回答简洁准确，不编造信息

"""

def generate(question: str, docs: list[dict]) -> str:
    context = "\n\n".join(f"[第{d['page']}页] {d['text']}" for d in docs)
    resp = _client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n资料：\n{context}"},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content

def generate_stream(question: str, docs: list[dict]):
    """流式生成：返回 stream 迭代器，逐块产出增量（新增）"""
    context = "\n\n".join(f"[第{d['page']}页] {d['text']}" for d in docs)
    return _client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n资料：\n{context}"},
        ],
        temperature=0.1,
        stream=True,                    # ← 唯一的区别
    )

async def generate_stream_async(question: str, docs: list[dict]):
    context = "\n\n".join(f"[第{d['page']}页] {d['text']}" for d in docs)
    stream = await _async_client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{question}\n\n资料：\n{context}"},
        ],
        temperature=0.1,
        stream=True,
    )
    async for chunk in stream:      # ← async 迭代，不阻塞事件循环
        yield chunk

def rewrite_query(query: str) -> str:
    resp = _client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "user", "content": f"""把下面的问题改写成适合法律条文检索的关键词，只输出关键词用空格分隔，必须用法言法语，不要输出其他内容。

问题：{query}
示例："16岁打工能独立签合同吗" → "十六周岁 未成年人 劳动收入 主要生活来源 完全民事行为能力 民事法律行为"""}],
        temperature=0,
    )
    return resp.choices[0].message.content