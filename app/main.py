from fastapi import FastAPI
from app.api import chat, health
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="RAG Service")
Instrumentator().instrument(app).expose(app)   # /metrics
app.include_router(chat.router)
app.include_router(health.router)