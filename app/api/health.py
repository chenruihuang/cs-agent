from fastapi import APIRouter
from app.rag.retriever import get_collection

router = APIRouter()

@router.get("/health")
def health():
    try:
        coll = get_collection()
        count = coll.count()
        return {"status": "ok", "chunks": count, "model": "loaded"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}