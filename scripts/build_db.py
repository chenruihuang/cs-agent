# 建库
import os,json
from scripts.parse_and_chunk import extract_text_from_pdf, build_chunks
from scripts.build_vector_db import build_vector_db
from pathlib import Path

if __name__ == "__main__":
    pages = extract_text_from_pdf(os.getenv("PDF_PATH"))
    chunks = build_chunks(pages)
    print(f"共生成 {len(chunks)} 个文本块")

    # 3. 统计
    lengths = [c["char_count"] for c in chunks]
    print(f"块大小：最小 {min(lengths)}，最大 {max(lengths)}，平均 {sum(lengths)//len(lengths)} 字")

    Path("chunks").mkdir(exist_ok=True)
    with open("chunks/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    build_vector_db()