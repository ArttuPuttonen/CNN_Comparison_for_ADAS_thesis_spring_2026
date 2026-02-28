FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /workspace

USER root

RUN apt-get update && apt-get install -y \
    libcudnn9-cuda-12 \
    libnccl2 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN pip install --no-cache-dir pandas numpy matplotlib

COPY train.py /workspace/train.py

CMD ["python", "train.py"]