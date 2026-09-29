import json,time
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.generator import generate, generate_stream, generate_stream_async,rewrite_query
from app.core.config import settings
from fastapi.responses import StreamingResponse


router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    pages: list[int]
    scores: list[float]

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    rewrite_ques = rewrite_query(req.message)
    docs = retrieve(rewrite_ques, top_k=settings.top_k_retrieve)
    top5 = rerank(rewrite_ques, docs, top_k=settings.top_k_rerank)
    answer = generate(rewrite_ques, top5)
    return ChatResponse(
        answer=answer,
        pages=[d["page"] for d in top5],
        scores=[round(d["rerank_score"], 3) for d in top5],
    )

async def event_stream(question: str):
    """异步生成器：一段段往外吐 SSE 消息"""
    # 1. 检索 + 重排（快，先算完，把来源页先发给前端）
    rewrite_ques = rewrite_query(question)
    docs = retrieve(rewrite_ques, top_k=settings.top_k_retrieve)
    top5 = rerank(rewrite_ques, docs, top_k=settings.top_k_rerank)
    yield f"data: {json.dumps({'type': 'docs', 'pages': [d['page'] for d in top5]}, ensure_ascii=False)}\n\n"

    # 2. async 流式生成（关键：async for，不用同步 for）
    async for chunk in generate_stream_async(rewrite_ques, top5):
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield f"data: {json.dumps({'type': 'token', 'content': delta.content}, ensure_ascii=False)}\n\n"

    # 3. 结束标记
    yield "data: [DONE]\n\n"

@router.post("/chat/stream", response_model=ChatResponse)
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        event_stream(req.message),
        media_type="text/event-stream",          # ← 告诉客户端这是 SSE
        headers={"X-Accel-Buffering": "no"},     # 反代理缓冲（nginx 场景防攒批）
    )
