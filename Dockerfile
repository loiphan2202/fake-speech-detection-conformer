FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Bước 1: cài torch CPU riêng (tránh pip kéo bản CUDA 8GB)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --timeout=300 \
        torch==2.0.1+cpu \
        torchaudio==2.0.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Bước 2: cài phần còn lại
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=300 -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]