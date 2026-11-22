# Jetson Orin Nano, JetPack 7.2 (L4T R39, CUDA 13). Build and run ON the Jetson.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA for arm64. Change the index if your CUDA/PyTorch combo differs.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN pip3 install torch torchvision --index-url ${TORCH_INDEX}
RUN pip3 install ultralytics requests

WORKDIR /app
COPY *.py ./
ENV DEVICE=cuda SHOW=0 YOLO_CONFIG_DIR=/app/.ultralytics
CMD ["python3", "detect.py"]
