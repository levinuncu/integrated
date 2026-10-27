# Drone Video and Data Transmission

## 1. Goal and hardware

We want to transmit live video from the drone to the base station. Image recognition runs on the drone, its results (e.g. person count, detected objects, body position and fire alerts) should be saved locally and sent to the base station when a connection is available. The video should also be recorded locally.

We already have or plan to use:
- **[Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/)** on the drone: It has no hardware H.264 encoder.
- **[Raspberry Pi Zero 2 W](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/)** at the base station: 2.4 GHz Wi-Fi only (802.11n), hardware H.264 decoding up to 1080p30.

## 2. Planned pipeline

```text
RGB camera
  -> Capture frames once
      -> Video branch: H.264 encode -> record locally
                                    -> live video stream -> Wi-Fi -> base station
      -> AI branch: resize/sample frames -> object recognition
                   -> JSON results + frame ID + timestamp
                   -> save locally -> send to base station when connected

Base station (Raspberry Pi Zero 2 W)
  -> Receive video without re-encoding
  -> Receive and store AI JSON messages
  -> Optionally forward video and results to another device
```

Video recording, streaming and AI recognition should run at the same time. We do not need to analyse every video frame. We should use independent processing queues so that slow AI inference does not block recording or streaming. [GStreamer `tee` documentation](https://gstreamer.freedesktop.org/documentation/coreelements/tee.html) explicitly recommends a separate `queue` for each branch.

**Offline behaviour:** Local storage is required on the drone. MQTT QoS alone does not ensure that newly generated results are delivered while the drone cannot reach the broker. We need a persistent local queue/database, message IDs, retry and acknowledgement logic. The live stream may stop during an outage, the recording must continue.

## 3. Possible transmission technologies

| Technology | Use and advantages | Risks / limitations |
|---|---|---|
| **[RTSP + RTP](https://www.rfc-editor.org/rfc/rfc7826)** | Widely supported live video; straightforward first prototype. RTSP controls the session; RTP carries the video. | TCP retransmission can delay newer video data after packet loss. RTSP by itself does not improve image quality. |
| **[RTP over UDP](https://www.rfc-editor.org/rfc/rfc3550)** | Simple, low-overhead real-time video transfer. | Lost packets are not automatically recovered by RTP; damaged or missing frames are possible. |
| **[SRT](https://github.com/Haivision/srt)** | Video over UDP with loss recovery by retransmission; useful on unstable wireless links. | Requires a latency buffer and extra bandwidth for retransmissions; delayed packets may still be dropped. [SRT latency documentation](https://github.com/Haivision/srt/blob/master/docs/features/latency.md). |

For AI results, we can use:
- **[MQTT](https://mqtt.org/)**: publish/subscribe, small JSON messages, QoS options; our preferred starting point. It needs a broker and application-level offline storage on the sender.
- **[ZeroMQ](https://zeromq.org/)**: brokerless messaging patterns; an alternative if we want more direct control over message transport.

**Initial choice:** RTSP/RTP over TCP + MQTT over ordinary Wi-Fi. Compare SRT + MQTT if packet loss affects the stream. A 2.4 GHz connection can be used for initial tests. If needed, try an external Wi-Fi access point with suitable antennas; for 5 GHz, the Zero 2 W needs an external adapter or a USB-Ethernet connection to the access point because its built-in Wi-Fi supports only 2.4 GHz.

## 4. Related papers and tested hardware

The papers below evaluate different parts of our problem. None validates the complete Pi 5 + Zero 2 W + AI + video system, so we need our own end-to-end measurements.

| Paper / source | Tested hardware / setup | Task and results |
|---|---|---|
| **[Fernandez et al. (2025)](https://doi.org/10.1109/ICMERR64601.2025.10949937)** | Aerial-robot field experiments; not a Pi 5-specific benchmark. | Compared RTP, RTSP, SRT and ROS-TCP for throughput, CPU usage, latency and packet integrity. In their tests, SRT/RTP showed advantages in network stability, while RTSP/ROS-TCP showed advantages in CPU efficiency. |
| **[Kilic et al. (2024)](https://doi.org/10.3390/drones8100551)** | NVIDIA Jetson Nano companion computer with hardware video encoding; WebRTC and Janus server. | Demonstrated video streaming and flight-data delivery for a browser-based multi-UAV ground station. Results cannot be transferred directly to Pi 5 software encoding. |
| **[Liu & Ahmed (2025)](https://doi.org/10.1109/WINCOM65874.2025.11313385)** | Raspberry Pi-based prototype; exact Pi model is not specified in the accessible abstract. | Compared MQTT and ZeroMQ for UAV data exchange, including transmission time, error rate and CPU load. Useful for our AI-result messages; no single protocol is automatically best for all workloads. |
| **[Bergesio et al. (2025)](https://doi.org/10.3390/electronics14030638)** | Raspberry Pi 5 and NVIDIA Jetson boards. | YOLOv8n with NCNN FP32: 118 ms/image in the model test on Pi 5. The tested Pi 5 pipeline did not meet the authors' real-time requirements. This is not a benchmark of simultaneous H.264 streaming. |
| **[Raspberry Pi Ltd. (2026), H.264 encoding performance](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010033-WP-1-H.264%20encoding%20performance%20on%20Raspberry%20Pi%205_series%20computers.pdf)** | Raspberry Pi 5, software `libx264`, 1080p30; manufacturer whitepaper, not peer-reviewed. | Low-latency configuration used as little as 60% of one CPU core; high-quality encoding required about 100–150% of one core. This does not include our AI workload. |

## 5. Risks and limitations

| Risk | Evidence and impact | Possible mitigation |
|---|---|---|
| **Video encoding uses significant CPU time.** | Pi 5 has no H.264 hardware encoder. The [Raspberry Pi whitepaper (2026)](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010033-WP-1-H.264%20encoding%20performance%20on%20Raspberry%20Pi%205_series%20computers.pdf) reports about 100–150% of one CPU core for high-quality 1080p30. This means one to one-and-a-half of four cores, not 100–150% of the whole Pi. | Measure the actual encoder preset and CPU load together with AI. Encode once if recording and streaming can share the same encoded output. |
| **AI and video compete for CPU and memory.** | [Bergesio et al. (2025)](https://doi.org/10.3390/electronics14030638) measured 118 ms/image for YOLOv8n NCNN FP32 on Pi 5. Running AI alongside video can reduce throughput further (to be measured). | Analyse fewer frames, use smaller models, consider a compatible AI accelerator, and use separate queues. |
| **Packet loss or retransmissions may interrupt video.** | The [aerial streaming comparison](https://doi.org/10.1109/ICMERR64601.2025.10949937) shows meaningful protocol differences. [SRT](https://github.com/Haivision/srt/blob/master/docs/features/latency.md) can recover losses using extra buffering, but this increases delay. | Compare RTSP/TCP and SRT on the same Wi-Fi link; reduce bitrate if necessary. |
| **Temperature and power may limit flight time.** | [Raspberry Pi's cooling tests](https://www.raspberrypi.com/news/heating-and-cooling-raspberry-pi-5/) show that sustained Pi 5 load can cause thermal throttling. The final drone power budget has not been measured. | Test sustained workload in the drone enclosure; check cooling, supply stability and battery drain. |
| **Data may be lost while disconnected or misaligned with the video.** | [MQTT session handling](https://mosquitto.org/man/mqtt-7.html) is not a replacement for a local record of results that have never reached the broker. Independently transmitted video and JSON can arrive at different times. | Durable local storage, IDs/timestamps, acknowledgements and retry after reconnect; match results to captured frames. |

## 6. First implementation and tests

We want to start with 1080p30 H.264 + RTSP/TCP for video and MQTT + JSON for AI results. A possible first set of test bitrates is 6, 10 and 15 Mbit/s (not guaranteed achievable values). We will record locally even when Wi-Fi fails.

Measure:
- Video only, AI only, then video + AI together: CPU, RAM, temperature, FPS and power.
- RTSP/TCP vs SRT with the same camera, bitrate and wireless setup.
- Disconnection and reconnection: whether all locally stored AI results eventually reach the base station, without duplicates in storage.

The first decision should be based on whether the Pi 5 can encode, record and run AI simultaneously. Only then should we decide whether changing the video transport or radio hardware is necessary.
