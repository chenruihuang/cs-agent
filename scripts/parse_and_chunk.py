# pdf解析并切分
import pymupdf
import json
import re
import os

from pathlib import Path
from dotenv import load_dotenv


CHUNK_SIZE = 500       # 每个块约 500 字（中文约 500 token）
CHUNK_OVERLAP = 100    # 块之间重叠 100 字，防止边界信息丢失
MIN_CHUNK_LEN = 50     # 过滤掉太短的碎片块

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """提取 PDF 文本，保留页码信息"""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        # 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    print(f"共提取 {len(pages)} 页文本")
    return pages

def build_chunks(pages: list[dict]) -> list[dict]:
    """对所有页切分，每个块记录来源页码"""
    all_chunks = []
    chunk_id = 0
    
    for page in pages:
        chunks = chunk_by_semantic_boundary(
            page["text"], CHUNK_SIZE, CHUNK_OVERLAP
        )
        for chunk_text in chunks:
            if len(chunk_text) >= MIN_CHUNK_LEN:
                all_chunks.append({
                    "id": f"chunk_{chunk_id:04d}",
                    "page": page["page"],
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                })
                chunk_id += 1
    
    return all_chunks

def chunk_by_semantic_boundary(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    智能切分：优先按段落/句子边界切，不硬切在句子中间。
    1. 先按段落分割（\n\n）
    2. 段落累加，超过 chunk_size 就切一块
    3. 每块保留 overlap 字的重叠
    """
    # 按段落分割
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # print(f"段落内容：{para} ")
        # 段落本身超长 → 二次切分
        # 如果当前块 + 新段落还没超限制，就累加
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        else:
            # 当前块已满，保存
            if current_chunk:
                chunks.append(current_chunk)
            # 新块从 overlap 开始（取上一块末尾 overlap 字）
            if overlap > 0 and chunks:
                current_chunk = chunks[-1][-overlap:] + "\n\n" + para
            else:
                current_chunk = para
            
            # 超长段落做切分
            if len(current_chunk) > chunk_size:
                sub_chunks = chunk_long_paragraph(current_chunk, chunk_size, overlap)

                # 除最后一块外，全部输出到 chunks
                chunks.extend(sub_chunks[:-1])

                # 最后一块留作 current_chunk，继续和后续段落拼接
                current_chunk = sub_chunks[-1] if sub_chunks else ""
                continue

    
    # 最后一块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def chunk_long_paragraph(para: str, chunk_size: int, overlap: int) -> list[str]:
    """把超长段落二次切分：优先按句子边界，退化按字符硬切"""
    # 1. 按中文句子结束符切（保留标点）
    sentences = re.split(r'(?<=[。！？；.!?])', para)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 2. 如果整段没有标点（切分失败），退化为字符硬切
    if len(sentences) <= 1:
        step = max(1, chunk_size - overlap)  # 保证相邻块有 overlap 重叠
        return [para[i:i + chunk_size] for i in range(0, len(para), step)]

    # 3. 按句子累加成块（逻辑和段落层一样）
    sub_chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current += sent
        else:
            if current:
                sub_chunks.append(current)
            if overlap > 0 and sub_chunks:
                current = sub_chunks[-1][-overlap:] + sent
            else:
                current = sent
    if current:
        sub_chunks.append(current)
    return sub_chunks

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
    
    # 5. 打印前 3 个块，人工检查切分质量
    print("\n===== 前 3 个块预览 =====")
    for c in chunks[:3]:
        print(f"\n[块 {c['id']} | 第{c['page']}页 | {c['char_count']}字]")
        print(c["text"][:200] + "...")