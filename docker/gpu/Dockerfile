FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MAX_JOBS=4

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
    wget

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install setuptools==69.5.1

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine && python setup.py install --force_cuda --blas=openblas
