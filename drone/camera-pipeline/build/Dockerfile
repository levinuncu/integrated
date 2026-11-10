FROM debian:bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Raspberry Pi package repository hinzufügen
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] https://archive.raspberrypi.com/debian/ bookworm main" \
    > /etc/apt/sources.list.d/raspberrypi.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-zmq \
    python3-gi \
    python3-gst-1.0 \
    python3-picamera2 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY camera.py cameras.py fake_cameras.py frame_bundle.py frame_publisher.py frame_synchronizer.py frame_wire.py main.py pipeline.py rolling_recording.py stream_destination.py ./

CMD ["python3", "main.py"]
