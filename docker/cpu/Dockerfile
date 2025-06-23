FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=4
ENV BLAS=openblas
ENV CPU_ONLY=1

WORKDIR /workspace
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    build-essential \
    libopenblas-dev \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    ninja-build \
    cmake \
    wget \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install setuptools==69.5.1

RUN pip install torch==1.12.1 torchvision==0.13.1

COPY requirements.txt .
RUN pip install -r requirements.txt


RUN pip install --no-cache-dir -U git+https://github.com/NVIDIA/MinkowskiEngine