FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV WANDB_MODE=offline

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    gnupg \
    lsb-release \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install gcsfuse
RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y gcsfuse && \
    apt-get clean

RUN pip install --upgrade pip

# Build Helpers
RUN pip install --no-cache-dir packaging ninja psutil wheel

# Install PyTorch
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install FlashAttention
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Copy requirements file
COPY geo-gpu.txt .

# Install dependencies
RUN pip install --no-cache-dir -r geo-gpu.txt

WORKDIR /app
COPY . /app

# ENTRYPOINT
ENTRYPOINT ["python3", "entrypoint.py"]