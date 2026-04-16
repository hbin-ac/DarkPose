FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.0.1 torchvision==0.3.0

# COCO API
RUN pip3 install pycocotools

# Copy project source
COPY . .

# Build NMS extension
RUN cd lib && make

# Default dataset and output mount points
VOLUME ["/data", "/app/output", "/app/log", "/app/models"]

ENTRYPOINT ["python3"]
CMD ["tools/train.py", "--help"]
