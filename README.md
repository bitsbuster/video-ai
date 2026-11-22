# Video AI: YOLO camera alerts on Jetson Orin Nano

A small app that runs in a Docker container on a Jetson Orin Nano. It reads a USB webcam, detects people, cats and dogs with YOLO (Ultralytics), and sends an alert with a photo to a Discord channel. It uses the NVIDIA GPU through CUDA.

> **Status:** this setup was written for JetPack 7.2 (Jetson Linux R39.2, CUDA 13). It is a first version and may need small changes, for example the PyTorch index in the `Dockerfile`.

## How it works

1. The app reads frames from the webcam.
2. YOLO detects people, cats and dogs in each frame.
3. When something appears, the app sends a message and a photo to Discord.
4. When nothing is seen for a few seconds, the app sends a "left" message.
5. Messages go through a queue and a background thread. This way, a slow network does not freeze the video.

The container runs without a window. All results go to Discord and to the container logs.

## 1. Requirements

- Jetson Orin Nano with JetPack 7.2. Check it with `cat /etc/nv_tegra_release` (it must show `R39`).
- Docker with the `nvidia` runtime. Check it with:

  ```bash
  docker info | grep -i runtime
  ```

- A USB webcam **connected to the Jetson**. The container only sees the cameras of the machine where it runs. Check it with:

  ```bash
  ls /dev/video*
  ```

- A Discord webhook URL (see section 2).

## 2. Create a Discord webhook

1. Open your Discord server.
2. Go to **Server Settings > Integrations > Webhooks**.
3. Click **New Webhook** and choose the channel.
4. Click **Copy Webhook URL**.

Save the URL in an environment variable. Do not write it in the code:

```bash
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
```

This variable only lives in the current terminal. To keep it, add the line to `~/.bashrc`.

> **Security:** anyone with this URL can post in your channel. Never commit it to GitHub. If you share it by mistake, delete the webhook in Discord and create a new one.

## 3. Build

```bash
git clone -b jetson-orin-nano https://github.com/bitsbuster/video-ai.git
cd video-ai
docker compose build
```

The first build downloads PyTorch and can take several minutes.

## 4. Check that the GPU is detected

```bash
docker compose run --rm video-ai python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- `True` and the GPU name: the GPU is available.
- `False`: see the Troubleshooting section.

## 5. Run the app

```bash
docker compose up
```

Run it in the background with `docker compose up -d`. See the logs with `docker compose logs -f`. Stop it with `docker compose down`.

The container restarts automatically if it stops (`restart: unless-stopped`).

## 6. Settings

Environment variables are set in `docker-compose.yml`:

| Name | Default | Meaning |
|---|---|---|
| `DISCORD_WEBHOOK` | none | URL of the Discord webhook. Required |
| `DEVICE` | `cuda` | `cuda` is the NVIDIA GPU, `cpu` is the CPU |
| `CAMERA` | `0` | Camera index (`/dev/video0` is `0`) |
| `SHOW` | `0` in the container | `1` opens a window. It needs a display, so keep `0` |

These values are at the top of `detect.py`:

| Name | Default | Meaning |
|---|---|---|
| `CLASSES` | person, cat, dog | Objects to detect (COCO class IDs 0, 15, 16) |
| `CONFIDENCE` | `0.50` | Minimum confidence, from 0 to 1. Higher means fewer false alerts |
| `EXIT_IN_SECONDS` | `3` | Seconds without detection before the "left" message |

To detect other objects, change `CLASSES`. The IDs come from the COCO dataset. For example, `2` is car.

## 7. Choose a model

The app uses `yolov8n.pt`. It downloads the first time you run it, so the container needs internet access.

| Model | Speed | Accuracy | Recommended use |
|---|---|---|---|
| `yolov8n.pt` | Very fast | Lower | Quick tests |
| `yolov8s.pt` | Fast | Medium | Good balance |
| `yolov8m.pt` | Medium | High | Better quality |

To change the model, change the name in `YOLO("...")`.

## 8. Troubleshooting

**`torch.cuda.is_available()` returns `False`, or a "no kernel image" error**
- The PyTorch wheel may not include the Orin GPU (`sm_87`). Change the index with `docker compose build --build-arg TORCH_INDEX=<url>`, or use a PyTorch image from NVIDIA NGC as the base image.
- Check that the `nvidia` runtime exists: `docker info | grep -i runtime`.

**`KeyError: 'DISCORD_WEBHOOK'`**
- The environment variable is not set in this terminal. Run the `export` command from section 2 again.

**The camera does not open**
- `/dev/video0` and `/dev/video1` are often the same camera (video and metadata). Set `CAMERA: "1"` in `docker-compose.yml`.
- If the camera uses another device, change the `devices` line, for example `/dev/video2:/dev/video0`.
- Check that the camera works on the Jetson: `ls /dev/video*`.

**No message arrives in Discord**
- Check the webhook URL.
- Look at the logs with `docker compose logs -f`. Errors are printed as `Error sending message`.
- Discord limits the number of messages. If the queue is full, the app prints `Queue is full, discarding message`.

**Model download is slow or blocked**
- Check your proxy or firewall. You can also download the `.pt` file by hand and put it in the project folder, then rebuild.

## 9. Project structure

```text
video-ai/
├── detect.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## 10. Faster inference with TensorRT

After everything works, export the model on the Jetson (the engine depends on the hardware):

```python
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="engine", half=True)
```

Then use `YOLO("yolov8n.engine")` in `detect.py`.

## 11. Next steps

- Use a video file instead of the webcam: `cv2.VideoCapture("video.mp4")`.
- Turn on object tracking: `model.track(frame, persist=True)`.
- Add a cooldown, so the app does not send many alerts in a short time.
