FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY svm_model.pkl /app/svm_model.pkl
COPY test_app.py /app/test_app.py

WORKDIR /app


CMD ["python", "app.py"]

