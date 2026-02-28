FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /workspace

# 🔴 TÄMÄ ON SE RATKAISEVA OSA
# TensorFlow GPU tarvitsee nämä user-space kirjastot
RUN apt-get update && apt-get install -y \
    libcudnn9-cuda-13 \
    libnccl2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python kirjastot
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib

# Kopioi training script
COPY train.py /workspace/train.py

CMD ["python", "train.py"]