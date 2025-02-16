FROM python:3-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    python3 \
    python3-dev \
    python3-setuptools \
    python3-distutils \
    build-essential \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#RUN pip install --no-cache-dir torch==2.5.0
#RUN pip install --no-cache-dir torchaudio

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --find-links /pip_cache

COPY . .
CMD ["python", "transcribe.py"]