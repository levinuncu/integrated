FROM debian:bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-zmq

WORKDIR /app

COPY frame_wire.py main.py ./

CMD ["python3", "main.py"]
