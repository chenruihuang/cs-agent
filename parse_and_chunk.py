# pdf解析并切分
import fitz


CHUNK_SIZE = 500       # 每个块约 500 字（中文约 500 token）
CHUNK_OVERLAP = 100    # 块之间重叠 100 字，防止边界信息丢失
MIN_CHUNK_LEN = 50     # 过滤掉太短的碎片块