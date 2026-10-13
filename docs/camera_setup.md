# Camera setup (Raspberry Pi 5)

## Hardware
- Raspberry Pi 5 with Raspberry Pi OS (64-bit)
- Raspberry Pi Camera Module 3 
- 22-to-15-pin camera cable 

## Connecting the camera
1. Power off the Pi and unplug it.
2. Lift the latch of a CAM/DISP connector, insert the narrow (22-pin) end of the cable, close the latch.
3. On the camera, insert the wide (15-pin) end with the contacts facing the camera PCB, close the latch.
4. Power on. Cameras are only detected at boot, so always reboot after reconnecting.


## Quick test
```bash
rpicam-hello --list-cameras          # should list the camera (e.g. imx708)
rpicam-hello -t 0                    # live preview, stop with Ctrl+C
rpicam-still -o test.jpg             # take a photo
rpicam-vid -t 10s -o test.mp4        # record 10 s of video
```

## Recording with Python (issue #10)
```bash
sudo apt install ffmpeg
python3 record_video.py -d 10        # saves to ./recordings/cam0_<timestamp>.mp4
python3 record_video.py --help       # all options
```

## Troubleshooting
| Symptom | Likely cause |
|---|---|
| `No cameras available` | Cable not seated / wrong orientation / camera plugged in after boot |
| Camera listed, but `Device timeout detected` | No image data over CSI: defective camera module or damaged cable |
| `command not found: rpicam-hello` | Not on Raspberry Pi OS, or `rpicam-apps` missing |

Background: our two old Camera Rev 1.3 modules (OV5647) were detected but never delivered frames.
Cross-checked on a Pi 3 and a Pi 4 with known-working setups, so they are defective.
The Pi 5 and both Arducam cables work fine with a new camera.
