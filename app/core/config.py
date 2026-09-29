# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    # 必填项：缺了启动直接报错，带清晰提示
    # 可选项
    deepseek_api_key: SecretStr
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    pdf_path: str = "data/test.pdf"
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    eval_path: str = "eval/questions.json"

# 全局单例
settings = Settings()
