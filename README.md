# Video AI: YOLO camera alerts on Apple Silicon Mac

A small test app. It reads your webcam, detects people, cats and dogs with YOLO (Ultralytics), and sends an alert with a photo to a Discord channel.

It is made for a new Mac with Apple Silicon (M1, M2, M3, M4...). It uses the Apple GPU through MPS.

It can also run in a Docker container on a Jetson Orin Nano with an NVIDIA GPU (CUDA). See section 12.

## How it works

1. The app reads frames from the webcam.
2. YOLO detects people, cats and dogs in each frame.
3. When something appears, the app sends a message and a photo to Discord.
4. When nothing is seen for a few seconds, the app sends a "left" message.
5. Messages go through a queue and a background thread. This way, a slow network does not freeze the video.

## 1. Requirements

- Mac with Apple Silicon (not Intel).
- A recent version of macOS.
- Python 3.9 or newer.
- Xcode command line tools.
- A Discord webhook URL (see section 5).

Check the architecture. The answer must be `arm64`:

```bash
uname -m
```

Install the command line tools (if you do not have them):

```bash
xcode-select --install
```

Check Python. If it is missing, install it with Homebrew (`brew install python`):

```bash
python3 --version
```

> **Important:** use a native `arm64` Python. If Python runs with Rosetta (x86_64), PyTorch cannot use the GPU. Check it with:
>
> ```bash
> python3 -c "import platform; print(platform.machine())"
> ```
>
> The result must be `arm64`.

## 2. Create the project and the virtual environment

```bash
mkdir video-ai
cd video-ai
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

When the environment is active, the prompt shows `(.venv)`.

## 3. Install the dependencies

```bash
pip install ultralytics opencv-python requests
```

This also installs PyTorch, NumPy and other needed libraries.

Save the versions, so you can repeat the install later:

```bash
pip freeze > requirements.txt
```

## 4. Check that the GPU is detected (MPS)

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

- `True`: the GPU is available.
- `False`: PyTorch will use the CPU. See the Troubleshooting section.

## 5. Create a Discord webhook

1. Open your Discord server.
2. Go to **Server Settings > Integrations > Webhooks**.
3. Click **New Webhook** and choose the channel.
4. Click **Copy Webhook URL**.

Save the URL in an environment variable. Do not write it in the code:

```bash
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
```

This variable only lives in the current terminal. To keep it, add the line to `~/.zshrc`.

> **Security:** anyone with this URL can post in your channel. Never commit it to GitHub. If you share it by mistake, delete the webhook in Discord and create a new one.

## 6. Run the app

```bash
python main.py
```

(Use the real name of your script file.)

A window called `cam` opens with the live video. Press `Esc` to quit.

The first time, macOS asks for camera permission. Click **Allow**.

## 7. Settings

These values are at the top of the script:

| Name | Default | Meaning |
|---|---|---|
| `CLASSES` | person, cat, dog | Objects to detect (COCO class IDs 0, 15, 16) |
| `DEVICE` | `mps` | Environment variable. `mps` is the Apple GPU, `cuda` is an NVIDIA GPU, `cpu` is the CPU |
| `CAMERA` | `0` | Environment variable. Camera index |
| `SHOW` | `1` | Environment variable. `0` runs without a window (needed in a container) |
| `CONFIDENCE` | `0.50` | Minimum confidence, from 0 to 1. Higher means fewer false alerts |
| `EXIT_IN_SECONDS` | `3` | Seconds without detection before the "left" message |
| `DISCORD_WEBHOOK` | none | Environment variable with the webhook URL |

To detect other objects, change `CLASSES`. The IDs come from the COCO dataset. For example, `2` is car.

## 8. Choose a model

The app uses `yolov8n.pt`. It downloads the first time you run it.

| Model | Speed | Accuracy | Recommended use |
|---|---|---|---|
| `yolov8n.pt` | Very fast | Lower | Quick tests |
| `yolov8s.pt` | Fast | Medium | Good balance |
| `yolov8m.pt` | Medium | High | Better quality |

To change the model, change the name in `YOLO("...")`.

## 9. Troubleshooting

**`mps.is_available()` returns `False`**
- Check that Python is `arm64` (see section 1).
- If the environment was created with an x86_64 Python, delete it (`rm -rf .venv`) and create it again with an arm64 Python.
- Update PyTorch: `pip install --upgrade torch torchvision`.

**`KeyError: 'DISCORD_WEBHOOK'`**
- The environment variable is not set in this terminal. Run the `export` command from section 5 again.

**The webcam does not open, or the screen is black**
- Go to System Settings > Privacy & Security > Camera.
- Give permission to the app that runs Python (Terminal, iTerm or your editor).
- Close the app and open it again.
- If you have more than one camera, try `cv2.VideoCapture(1)`.

**No message arrives in Discord**
- Check the webhook URL.
- Look at the terminal. Errors are printed as `Error sending message`.
- Discord limits the number of messages. If the queue is full, the app prints `Queue is full, discarding message`.

**Error with an operation on MPS**
- Some PyTorch operations are not supported on MPS yet. Let PyTorch use the CPU for them:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**Model download is slow or blocked**
- Check your proxy or firewall. You can also download the `.pt` file by hand and put it in the project folder.

## 10. Project structure

```text
video-ai/
├── .venv/
├── main.py
├── requirements.txt
└── yolov8n.pt
```

## 11. Next steps

- Use a video file instead of the webcam: `cv2.VideoCapture("video.mp4")`.
- Turn on object tracking: `model.track(frame, persist=True)`.
- Add a cooldown, so the app does not send many alerts in a short time.
- Export the model to CoreML for better speed on Mac: `model.export(format="coreml")`.

## 12. Run on a Jetson Orin Nano (Docker)

The app can run in a container on a Jetson Orin Nano. It uses the NVIDIA GPU through CUDA and runs without a window.

> **Status:** this setup was written for JetPack 7.2 (Jetson Linux R39.2, CUDA 13). It is a first version and may need small changes, for example the PyTorch index in the `Dockerfile`.

### Requirements

- Jetson Orin Nano with JetPack 7.2. Check it with `cat /etc/nv_tegra_release` (it must show `R39`).
- Docker with the `nvidia` runtime. Check it with `docker info | grep -i runtime`.
- A USB webcam **connected to the Jetson**. The container only sees cameras of the machine where it runs. Check it with `ls /dev/video*`.
- A Discord webhook URL (see section 5).

### Build and run

```bash
git clone -b jetson-orin-nano <your-repo-url>
cd video-ai
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
docker compose build
```

Check that PyTorch sees the GPU:

```bash
docker compose run --rm video-ai python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Start the app:

```bash
docker compose up
```

Run it in the background with `docker compose up -d`. Stop it with `docker compose down`.

### Jetson troubleshooting

**`torch.cuda.is_available()` returns `False`, or "no kernel image" error**
- The PyTorch wheel may not include the Orin GPU (`sm_87`). Change the index with `docker compose build --build-arg TORCH_INDEX=<url>`, or use a PyTorch image from NVIDIA NGC as the base image.

**The camera does not open**
- `/dev/video0` and `/dev/video1` are often the same camera (video and metadata). Set `CAMERA: "1"` in `docker-compose.yml`.
- Use another device with `devices: - /dev/video1:/dev/video0` if needed.

**Model download fails**
- The container needs internet access the first time to download `yolov8n.pt`.

### Faster inference with TensorRT

After it works, export the model on the Jetson (the engine depends on the hardware):

```python
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="engine", half=True)
```

Then use `YOLO("yolov8n.engine")` in `detect.py`.
