# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 必填项：缺了启动直接报错，带清晰提示
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    pdf_path: str = "data/test.pdf"
    # 可选项
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

# 全局单例
settings = Settings()
