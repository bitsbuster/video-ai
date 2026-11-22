# Video AI: YOLO camera alerts on Jetson Orin Nano (Tapo camera)

A small app that runs in a Docker container on a Jetson Orin Nano. It reads the video of a Tapo IP camera (RTSP), detects people, cats and dogs with YOLO (Ultralytics), and sends an alert with a photo to a Discord channel. It uses the NVIDIA GPU through CUDA.

> **Status:** this setup was written for JetPack 7.2 (Jetson Linux R39.2, CUDA 13). It is a first version and may need small changes, for example the PyTorch index in the `Dockerfile`.

## How it works

1. The app reads frames from the Tapo camera.
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

- A Tapo camera on the same network as the Jetson, with a camera account (see section 2).
- A Discord webhook URL (see section 3).

## 2. Set up the Tapo camera

1. In the Tapo app, open the camera: **Settings > Advanced Settings > Camera Account**. Create a username and a password. This account is **not** your Tapo cloud account.
2. Find the camera IP: Tapo app > **Settings > Device Info**. Give the camera a fixed IP in your router, so it does not change.
3. The stream URL is:

   ```text
   rtsp://USER:PASSWORD@CAMERA_IP:554/stream1
   ```

   `stream1` is high quality and `stream2` is lower quality (lighter and faster). If the password has special characters (`@`, `:`, `/`, `#`), replace them with their URL code (for example `@` is `%40`).
4. Test the URL from the Jetson, before using Docker: `ffplay "rtsp://USER:PASSWORD@CAMERA_IP:554/stream1"` (or open it in VLC).

## 3. Create a Discord webhook

1. Open your Discord server.
2. Go to **Server Settings > Integrations > Webhooks**.
3. Click **New Webhook** and choose the channel.
4. Click **Copy Webhook URL**.

Save the webhook and the camera URL in a `.env` file in the project folder. Docker Compose reads it automatically. Do not write them in the code:

```bash
cat > .env <<'EOF'
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
CAMERA=rtsp://USER:PASSWORD@CAMERA_IP:554/stream1
EOF
chmod 600 .env
```

`.env` is in `.gitignore`, so it is not uploaded to GitHub.

> **Security:** anyone with the webhook URL can post in your channel, and anyone with the camera URL can see your camera. Never commit it to GitHub. If you share it by mistake, delete the webhook in Discord and create a new one.

## 4. Build

```bash
git clone -b jetson-orin-nano https://github.com/bitsbuster/video-ai.git
cd video-ai
docker compose build
```

The first build downloads PyTorch and can take several minutes.

## 5. Check that the GPU is detected

```bash
docker compose run --rm video-ai python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- `True` and the GPU name: the GPU is available.
- `False`: see the Troubleshooting section.

## 6. Run the app

```bash
docker compose up
```

Run it in the background with `docker compose up -d`. See the logs with `docker compose logs -f`. Stop it with `docker compose down`.

The container restarts automatically if it stops (`restart: unless-stopped`).

## 7. Settings

Environment variables are set in `docker-compose.yml`:

| Name | Default | Meaning |
|---|---|---|
| `DISCORD_WEBHOOK` | none | URL of the Discord webhook. Required |
| `DEVICE` | `cuda` | `cuda` is the NVIDIA GPU, `cpu` is the CPU |
| `CAMERA` | none | Camera stream URL (`rtsp://...`). A number (`0`) opens a USB webcam, but the compose file does not pass `/dev/video0` |
| `SHOW` | `0` in the container | `1` opens a window. It needs a display, so keep `0` |

These values are at the top of `detect.py`:

| Name | Default | Meaning |
|---|---|---|
| `CLASSES` | person, cat, dog | Objects to detect (COCO class IDs 0, 15, 16) |
| `CONFIDENCE` | `0.50` | Minimum confidence, from 0 to 1. Higher means fewer false alerts |
| `EXIT_IN_SECONDS` | `3` | Seconds without detection before the "left" message |

To detect other objects, change `CLASSES`. The IDs come from the COCO dataset. For example, `2` is car.

## 8. Choose a model

The app uses `yolov8n.pt`. It downloads the first time you run it, so the container needs internet access.

| Model | Speed | Accuracy | Recommended use |
|---|---|---|---|
| `yolov8n.pt` | Very fast | Lower | Quick tests |
| `yolov8s.pt` | Fast | Medium | Good balance |
| `yolov8m.pt` | Medium | High | Better quality |

To change the model, change the name in `YOLO("...")`.

## 9. Troubleshooting

**`torch.cuda.is_available()` returns `False`, or a "no kernel image" error**
- The PyTorch wheel may not include the Orin GPU (`sm_87`). Change the index with `docker compose build --build-arg TORCH_INDEX=<url>`, or use a PyTorch image from NVIDIA NGC as the base image.
- Check that the `nvidia` runtime exists: `docker info | grep -i runtime`.

**`KeyError: 'DISCORD_WEBHOOK'`**
- `DISCORD_WEBHOOK` is not set. Check that the `.env` file exists in the project folder (section 3) and recreate the container with `docker compose up -d --force-recreate`.

**The camera does not open (`Cannot open the camera`)**
- Check the camera account, the IP and the URL with `ffplay` from the Jetson (section 2).
- Check that the Jetson and the camera are on the same network and that the camera is on.
- If the video is choppy, use `stream2` (lower quality).

**No message arrives in Discord**
- Check the webhook URL.
- Look at the logs with `docker compose logs -f`. Errors are printed as `Error sending message`.
- Discord limits the number of messages. If the queue is full, the app prints `Queue is full, discarding message`.

**Model download is slow or blocked**
- Check your proxy or firewall. You can also download the `.pt` file by hand and put it in the project folder, then rebuild.

## 10. Project structure

```text
video-ai/
├── detect.py
├── Dockerfile
├── .gitignore
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## 11. Faster inference with TensorRT

After everything works, export the model on the Jetson (the engine depends on the hardware):

```python
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="engine", half=True)
```

Then use `YOLO("yolov8n.engine")` in `detect.py`.

## 12. Next steps

- Use a video file instead of the camera: `cv2.VideoCapture("video.mp4")`.
- Turn on object tracking: `model.track(frame, persist=True)`.
- Add a cooldown, so the app does not send many alerts in a short time.
