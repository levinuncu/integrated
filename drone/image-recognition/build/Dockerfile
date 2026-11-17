
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-posture.txt prepare_posture.py ./
RUN pip install --no-cache-dir -r requirements-posture.txt

COPY detector.py dfire.yaml frame_wire.py \
     image_posture.py main.py record_buffer.py ./

# Pretrained models for offline inference
# COPY models/ ./models/

CMD ["python3", "main.py", "--download-models", "--output-dir", "/app/data/imr_records"]
