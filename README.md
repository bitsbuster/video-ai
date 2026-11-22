# Video AI: YOLO camera alerts on Jetson Orin Nano (Tapo camera)

A small app that runs in a Docker container on a Jetson Orin Nano. It reads the video of a Tapo IP camera (RTSP), detects people, cats and dogs with YOLO (Ultralytics), and sends an alert with a photo to a Discord channel. It uses the NVIDIA GPU through CUDA.

Tested on a Jetson Orin Nano with JetPack 7.2 (Jetson Linux R39.2.1, CUDA 13.2).

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

1. In the Tapo app, open the camera: **Settings > Advanced Settings > Camera Account**. Turn it on and create a username and a password. This account is **not** your Tapo cloud account. Use a password with only letters and numbers.
2. Find the camera IP: Tapo app > **Settings > Device Info**. Give the camera a fixed IP in your router, so it does not change.
3. The stream URL is:

   ```text
   rtsp://USER:PASSWORD@CAMERA_IP:554/stream1
   ```

   `stream1` is high quality and `stream2` is lower quality (lighter and faster). If the password has special characters (`@`, `:`, `/`, `#`), replace them with their URL code (for example `@` is `%40`).

## 3. Create a Discord webhook

1. Open your Discord server.
2. Go to **Server Settings > Integrations > Webhooks**.
3. Click **New Webhook** and choose the channel.
4. Click **Copy Webhook URL**.

## 4. Get the code and set the secrets

```bash
git clone -b jetson-tapo https://github.com/bitsbuster/video-ai.git
cd video-ai
```

Save the webhook and the camera URL in a `.env` file in the project folder. Docker Compose reads it automatically. Do not write them in the code:

```bash
cat > .env <<'EOF'
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
CAMERA=rtsp://USER:PASSWORD@CAMERA_IP:554/stream1
EOF
chmod 600 .env
```

`.env` is in `.gitignore`, so it is not uploaded to GitHub.

> **Security:** anyone with the webhook URL can post in your channel, and anyone with the camera URL can see your camera. Never commit them to GitHub. If you share the webhook by mistake, delete it in Discord and create a new one. If you share the camera password, change it in the Tapo app.
>
> When OpenCV cannot open a stream, it can print the full URL, including the password, in the logs. Hide the password before you paste logs anywhere.

## 5. Build

```bash
docker compose build
```

The first build downloads PyTorch and can take several minutes. Later builds are fast, because Docker keeps the layers in its cache.

## 6. Check the GPU and the camera

Check that PyTorch sees the GPU:

```bash
docker compose run --rm video-ai python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It must print `True` and the GPU name. PyTorch may also print a warning that your GPU (compute capability 8.7) is not in its list of supported GPUs. In our tests the GPU worked anyway.

Check that the Jetson reaches the camera:

```bash
ping -c 3 CAMERA_IP
nc -vz CAMERA_IP 554
```

`nc` must say `succeeded`. Then check that the video opens. This asks for the user and the password without saving them in the shell history, and hides the password in the output:

```bash
read -rp "User: " U
read -rsp "Password: " P; echo
docker run --rm --network host \
  -e CAMERA="rtsp://$U:$P@CAMERA_IP:554/stream1" \
  -e OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" \
  video-ai-video-ai \
  python3 -c "import os,cv2; c=cv2.VideoCapture(os.environ['CAMERA'], cv2.CAP_FFMPEG); print('opened:', c.isOpened()); print('frame:', c.read()[0])" 2>&1 | sed "s/$P/XXXX/g"
unset P
```

It must print `opened: True` and `frame: True`. The image name is `<folder>-video-ai`. If your folder is `video-ai`, it is `video-ai-video-ai` (see `docker images`).

## 7. Run the app

```bash
docker compose up -d
docker compose logs -f
```

The first start downloads `yolov8n.pt` (about 6 MB), so the container needs internet access. If you recreate the container, it downloads the file again.

The logs show nothing until something is detected. Stand in front of the camera: you should see `person detected: person: 1` and receive the photo in Discord. About 3 seconds after you leave, the app sends `person left`.

Stop it with `docker compose down`. The container restarts automatically if it stops (`restart: unless-stopped`).

To check that the GPU is used, run `sudo tegrastats` while the app runs. The value `GR3D_FREQ` goes up when YOLO processes a frame.

## 8. Settings

Environment variables are set in `docker-compose.yml` and `.env`:

| Name | Default | Meaning |
|---|---|---|
| `DISCORD_WEBHOOK` | none | URL of the Discord webhook. Required |
| `CAMERA` | none | URL of the camera stream (`rtsp://...`). Required |
| `DEVICE` | `cuda` | `cuda` is the NVIDIA GPU, `cpu` is the CPU |
| `SHOW` | `0` in the container | `1` opens a window. It needs a display, so keep `0` |

These values are at the top of `detect.py`:

| Name | Default | Meaning |
|---|---|---|
| `CLASSES` | person, cat, dog | Objects to detect (COCO class IDs 0, 15, 16) |
| `CONFIDENCE` | `0.50` | Minimum confidence, from 0 to 1. Higher means fewer false alerts |
| `EXIT_IN_SECONDS` | `3` | Seconds without detection before the "left" message |

To detect other objects, change `CLASSES`. The IDs come from the COCO dataset. For example, `2` is car. After you change `detect.py`, run `docker compose up -d --build`.

## 9. Troubleshooting

**`torch.cuda.is_available()` returns `False` and `device_count` is 0**
- The container does not see the GPU. Check that `docker-compose.yml` has `runtime: nvidia` and the variables `NVIDIA_VISIBLE_DEVICES: all` and `NVIDIA_DRIVER_CAPABILITIES: all`. Without them, the container started from the `ubuntu` image does not get the GPU.
- Check that the `nvidia` runtime exists: `docker info | grep -i runtime`.

**A "no kernel image" error**
- The PyTorch wheel does not include the Orin GPU. Change the index with `docker compose build --build-arg TORCH_INDEX=<url>`, or use a PyTorch image from NVIDIA NGC as the base image.

**The container restarts all the time**
- Look at the reason: `docker compose logs --tail 40`.

**`variable is not set` or `KeyError: 'DISCORD_WEBHOOK'`**
- Check that the `.env` file is in the same folder as `docker-compose.yml` and has both lines. Then run `docker compose up -d --force-recreate`.

**The camera does not open (`Cannot open the camera`)**
- Run the checks of section 6: `ping`, `nc` and the video test. If `ping` or `nc` fail, the problem is the network: check that the Jetson and the camera are on the same network, and the camera IP.
- `401 Unauthorized`: the user or the password is wrong. The user is case-sensitive, and **Camera Account** must be turned on in the Tapo app.
- `can't create capture`, `400 Bad Request` or `CSeq ... expected, ... received`: the camera refuses the stream. Close the live view in the Tapo app and stop the container (`docker compose stop`). Then restart the camera (unplug it for 10 seconds), wait a minute and try once. Tapo cameras accept only a few RTSP connections at the same time, and many failed tests can keep them busy.
- If the video is choppy or the CPU is high, use `stream2` (lower quality).

**No message arrives in Discord**
- Check the webhook URL.
- Look at the logs with `docker compose logs -f`. Errors are printed as `Error sending message`.
- Discord limits the number of messages. If the queue is full, the app prints `Queue is full, discarding message`.

**Model download is slow or blocked**
- Check your proxy or firewall.

## 10. Project structure

```text
video-ai/
├── detect.py            the app
├── Dockerfile           image with PyTorch (CUDA) and Ultralytics
├── docker-compose.yml   runs the app with the NVIDIA runtime
├── .dockerignore
├── .gitignore           ignores .env
└── README.md
```
