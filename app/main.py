from fastapi import FastAPI
from app.api import chat, health

app = FastAPI(title="RAG Service")
app.include_router(chat.router)
app.include_router(health.router)