

def rewrite_query(query: str) -> str:
    resp = _client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "user", "content": f"""把下面的问题改写成适合法律条文检索的关键词，只输出关键词用空格分隔，必须用法言法语，不要输出其他内容。

问题：{query}
示例："16岁打工能独立签合同吗" → "十六周岁 未成年人 劳动收入 主要生活来源 完全民事行为能力 民事法律行为"""}],
        temperature=0,
    )
    return resp.choices[0].message.content