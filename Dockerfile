FROM python:3.12-slim

# 系统依赖：torch / onnxruntime 需要（slim 缺 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 文件名必须和实际下载的一致（COPY 原名，别改）
# 先用curl.exe -OJ "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.12.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" 命令下载到本地
COPY torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/pip pip install /tmp/torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
RUN python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 先设全局镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 先装依赖（利用层缓存：改代码不会触发重新装依赖）
# RUN --mount=type=cache,target=/root/.cache/pip pip install --no-index --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ torch
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt

# 拷代码 —— 不拷 .env，不拷 data（运行时注入/挂载）
COPY *.py .

# 模型下载走国内镜像
ENV HF_ENDPOINT=https://hf-mirror.com

# 一次性建库任务（配合 docker compose run --rm 使用）
CMD ["sh", "-c", "python parse_and_chunk.py && python build_vector_db.py"]
