#!/usr/bin/env bash
# Set up EdiVal + the GUI (app.py) inside WSL2 Ubuntu 22.04 on a Windows PC with an NVIDIA GPU.
# Normally launched by env_setup/setup_windows.ps1; safe to re-run (finished steps are skipped).
#
# Differences from setup_edival.sh:
#   - installs vLLM first and compiles GroundingDINO against the torch that vLLM brings in
#     (building it before vLLM upgrades torch leaves a broken CUDA extension),
#   - installs the CUDA 12.4 toolkit for WSL (needed to compile GroundingDINO), Miniforge, and gradio.
set -eo pipefail  # no -u: conda's activate scripts reference unset variables

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="$HOME/miniforge3"
ENV_NAME=edival
CUDA_HOME=/usr/local/cuda-12.4

step() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

step "Checking the GPU is visible inside WSL"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi does not work inside WSL. Install/update the NVIDIA driver on Windows (not inside Ubuntu) and run 'wsl --update'." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step "Installing system packages (you may be asked for your Ubuntu password)"
sudo apt-get update -y
sudo apt-get install -y git wget build-essential libgl1 libglib2.0-0

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  step "Installing CUDA 12.4 toolkit for WSL (~3 GB, compiler only - the driver comes from Windows)"
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update -y
  sudo apt-get install -y cuda-toolkit-12-4
fi

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  step "Installing Miniforge (conda)"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/bin/activate"

if [ ! -d "$CONDA_DIR/envs/$ENV_NAME" ]; then
  step "Creating conda env '$ENV_NAME' (Python 3.10)"
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

step "Installing Python packages (vLLM + torch first, ~10 GB of downloads)"
pip install --upgrade pip
pip install vllm==0.8.4
pip install pillow matplotlib scipy scikit-image gdown \
  supervision==0.22.0 addict yapf timm==0.9.16 pycocotools pandas tabulate \
  diffusers tensorboard openai datasets accelerate "transformers==4.57.6" gradio

step "Compiling GroundingDINO against the installed torch"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
# Compile only for this GPU's architecture (much faster), e.g. 8.9 for an RTX 4090.
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
cd "$REPO_DIR"
pip install -e ./GroundingDINO --no-build-isolation --config-settings editable_mode=compat
mkdir -p GroundingDINO/weights
if [ ! -f GroundingDINO/weights/groundingdino_swint_ogc.pth ]; then
  wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -O GroundingDINO/weights/groundingdino_swint_ogc.pth
fi

step "Checking the install"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
from groundingdino import _C  # fails if the CUDA extension did not build
import vllm, gradio
print(f"OK: torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}, vllm {vllm.__version__}, gradio {gradio.__version__}")
EOF

step "Hugging Face login"
if python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
  echo "Already logged in to Hugging Face."
else
  cat <<'EOF'
The consistency model (facebook/dinov3-vitb16-pretrain-lvd1689m) is gated on Hugging Face:
  1. Create a free account at https://huggingface.co/join
  2. Open https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m and accept the license
  3. Create a "Read" token at https://huggingface.co/settings/tokens and paste it below
EOF
  python -c "from huggingface_hub import login; login()"
fi

step "Done"
echo "Start the GUI with the 'Start EdiVal GUI' shortcut on your Windows desktop."
