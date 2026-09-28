# 建库
import os
from app.scripts.parse_and_chunk import extract_text_from_pdf, build_chunks
from app.scripts.build_vector_db.py import build_vector_db

if __name__ == "__main__":
    pages = extract_text_from_pdf(os.getenv("PDF_PATH"))
    chunks = build_chunks(pages)
    print(f"共生成 {len(chunks)} 个文本块")

    build_vector_db()