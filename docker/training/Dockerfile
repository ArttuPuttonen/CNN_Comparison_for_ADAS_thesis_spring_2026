FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

CMD ["python", "train.py"]