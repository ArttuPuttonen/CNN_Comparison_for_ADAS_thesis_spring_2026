FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libcudnn9-cuda-13 \
    libnccl2

# Python-kirjastot
RUN pip install --no-cache-dir pandas numpy matplotlib

COPY train.py /workspace/train.py

CMD ["python", "train.py"]