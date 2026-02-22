FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /workspace

RUN pip install --no-cache-dir pandas numpy matplotlib

COPY train.py /workspace/train.py

CMD ["python", "train.py"]