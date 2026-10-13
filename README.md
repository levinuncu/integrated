## Code Structure

- `/base_station/`: Contains the applications running on the base station.
- `/drone/`: Contains the applications running on the drone.

## Base Station

- `base_station/video-receiver/`: Receives the H.264 video stream from the drone via RTP/UDP, stores the last 10 minutes of video locally in a rolling buffer, and forwards the stream to MediaMTX for WebRTC playback.
- `base_station/mediamtx.yml`: Configures the MediaMTX server to receive the forwarded video stream and make it available to browsers and other devices via WebRTC.

### Drone

- `/drone/camera-pipeline/`: Captures RGB camera frames and thermal sensor data. The RGB stream is split into a video branch for H.264 encoding, local 10-minute rolling recording, and RTP/UDP streaming to the base station. The AI branch resizes and samples RGB frames, synchronizes them with thermal data, and sends the combined frames to the image-recognition service via ZeroMQ.
- `/drone/image-recognition/`: Receives synchronized RGB frames and thermal data via ZeroMQ. Runs image-recognition models to detect people, body positions, movement, fire, and potential heat sources. Produces structured detection results containing frame IDs and timestamps for further processing.

## Local Testing

```sh
docker compose -f base_station/compose.yaml up --build -d
docker compose -f drone/compose.yaml up --build -d
```

Open: http://localhost:8889/drone/
