FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
RUN python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

# ===== 重构后：拷贝包结构，不再用 *.py =====
COPY app/ ./app/
# 建库/评估脚本（build-db 用），容器内调用python scripts.parse_and_chunk && python scripts.build_vector_db
COPY scripts/ ./scripts/

ENV HF_ENDPOINT=https://hf-mirror.com

# 默认启动 FastAPI —— 常驻服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
