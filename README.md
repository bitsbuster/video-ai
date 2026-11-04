Testing Video AI

mkdir video-ai
cd video-ai
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python

For mac, test if detects GPU

python -c "import torch; print(torch.backends.mps.is_available())"

