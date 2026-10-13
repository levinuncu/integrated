# On-board Image Recognition

## 1. Goal and hardware

We want to record video on the drone and detect people, their body position (standing or lying), movement, fire and possible heat sources. The drone should save the results and send them to the base station when a connection is available.

We already have:
- [Raspberry Pi Camera Module 3 Wide](https://www.raspberrypi.com/products/camera-module-3/): RGB camera, 12 MP, 120° field of view (diagonal), about 4 g.
- [MLX90640BAB](https://www.melexis.com/en/product/mlx90640/far-infrared-ir-sensor-array): ir sensor, 32 × 24 temperature values, 55° × 35° field of view, I²C.


## 2. Planned pipeline

```text
RGB camera
  -> Record and save video
  -> Resize and prepare frames for AI
  -> YOLO26n: detect people (and fire with a trained model)
  -> YOLO26n-pose: estimate body position when needed
  -> Optional ByteTrack: count people and track movement

MLX90640BAB
  -> Read temperature values
  -> Detect possible heat sources

AI results + ir data + timestamps
  -> Create JSON result
  -> Save locally and send to base station if connected
```

Video recording and AI processing should run at the same time. We do not need to run AI on every recorded frame. We will first try to match RGB and ir results by time.

## 3. Possible software and datasets

- **[YOLO26n](https://docs.ultralytics.com/models/yolo26/)** for detecting people.
`n` means nano: this is the smallest YOLO26 detection variant. We want to start with a small model because the SBC has limited computing power. The pretrained model can already detect the `person` class using [COCO](https://docs.ultralytics.com/datasets/detect/coco/). Performance on our SBC still needs to be measured.
- **Fire detection:** Fire is not a standard COCO class. We need to train or fine-tune a YOLO model with fire images. A possible dataset is [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) (fire and smoke bounding boxes). We must test it with images from our drone, not only the dataset.
- **[YOLO26n-pose](https://docs.ultralytics.com/tasks/pose/)** for body keypoints. We can use the keypoints to estimate `standing` or `lying`. These labels are not direct outputs of the pretrained pose mode, so we need to define and test the rules ourselves.
- **Optional [ByteTrack](https://docs.ultralytics.com/modes/track/)** to keep an ID for each detected person across frames. This may help with counting and movement detection. Movement is harder to measure because the drone itself moves.

Example output:

```json
{
  "time_ms": 123456,
  "person_count": 1,
  "persons": [
    {
      "id": 1,
      "confidence": 0.94,
      "posture": "lying",
      "moving": false
    }
  ],
  "fire_detected": false,
  "fire_confidence": null,
  "ir_hotspot": true
}
```

## 4. Related papers and tested hardware
Next a comparison of three papers in the same area of research, so that we can derive hardware requirements for SBCs.

| Paper | Tested SBC / specs | Task and results |
|---|---|---|
| [Alves et al. (2023)](https://doi.org/10.3390/s23239403) | **Orange Pi 5:** RK3588S, 4× A76 + 4× A55, 8 GB RAM; but uses MLX90640BAA | Thermal-assisted MediaPipe pose detection; **19 FPS** with a person; **6.99 W** for pose detection alone. Thermal-controlled processing used less power in the study. |
| [Alves et al. (2023)](https://doi.org/10.3390/s23239403) | **Raspberry Pi 4B:** 4× Cortex-A72, 8 GB RAM | Same pose task; **5 FPS** with a person; **6.27 W** for pose detection alone. |
| [YOLO on edge devices for drones (2025)](https://doi.org/10.3390/electronics14030638) | **Raspberry Pi 5:** 4× Cortex-A76, 8 GB RAM | YOLOv8n with NCNN (FP32): **118 ms per image** in the model test. The full tested drone pipeline was not stable on this SBC. |
| [FCMI-YOLO (2025)](https://doi.org/10.1371/journal.pone.0329555) | **Orange Pi 5 Plus:** RK3588, built-in 6-TOPS NPU; RAM size not reported in the paper | Fire detection with an IMX577 camera: **23.4 FPS** with the authors' optimized model and parallel processing. |
