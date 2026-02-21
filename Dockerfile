FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-distutils python3.9-venv \
    git curl ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 point to python3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 \
    && update-alternatives --set python3 /usr/bin/python3.9 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

RUN python3 -m pip install --upgrade pip setuptools

# Install PyTorch matching CUDA 11.3
RUN python3 -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install mmcv-full matching torch 1.10 + cu113
RUN python3 -m pip install mmcv-full==1.5.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# Install remaining requirements (excluding torch/mmcv which are already installed)
COPY requirements.txt /tmp/requirements.txt
RUN python3 - <<'PY'
from pathlib import Path
req = Path('/tmp/requirements.txt').read_text().splitlines()
filtered = []
for line in req:
    if line.startswith('torch==') or line.startswith('torchvision==') or line.startswith('mmcv-full=='):
        continue
    if line.strip():
        filtered.append(line)
Path('/tmp/requirements.filtered.txt').write_text('\n'.join(filtered))
PY
RUN python3 -m pip install -r /tmp/requirements.filtered.txt

WORKDIR /workspace
