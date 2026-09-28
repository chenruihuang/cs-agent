from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.rag.generator import generate
from app.core.config import settings

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    pages: list[int]
    scores: list[float]

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    docs = retrieve(req.message, top_k=settings.top_k_retrieve)
    top5 = rerank(req.message, docs, top_k=settings.top_k_rerank)
    answer = generate(req.message, top5)
    return ChatResponse(
        answer=answer,
        pages=[d["page"] for d in top5],
        scores=[round(d["rerank_score"], 3) for d in top5],
    )
