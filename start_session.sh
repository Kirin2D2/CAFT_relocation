#!/bin/bash
apt-get update -qq && apt-get install -y -qq tmux
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/relocation
python -m venv venv 2>/dev/null
source venv/bin/activate
export HF_HOME=/workspace/hf

# This pod's driver is 570.x / CUDA 12.8. The default PyPI torch wheel is built
# against cu130 and silently gives torch.cuda.is_available() == False here --
# no error, just a model that lands on CPU. Pin the cu128 build and install it
# FIRST so nothing below can drag cu130 back in. torchvision must match too:
# without it in the venv, transformers picks up the system dist-packages build
# and dies with "operator torchvision::nms does not exist". Same for torchaudio
# (system ships 2.8.0+cu128 linked against system torch 2.8.0).
# Also install ipython INTO the venv so the kernel does not put system
# dist-packages on sys.path in the first place.
pip install -q "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128

# fsspec pinned: datasets 5.0.1 requires <=2026.6.0, torch's deps pull newer.
pip install -q ipython transformers peft accelerate datasets scikit-learn matplotlib bitsandbytes "fsspec<=2026.6.0"

# Fail loudly rather than silently training on CPU.
python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA UNAVAILABLE -- torch build does not match driver"
print(f"cuda ok: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
PYEOF
echo "ready"
