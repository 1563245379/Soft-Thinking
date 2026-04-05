#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pip install -r requirements.txt

cd sglang_soft_thinking_pkg
pip install -e "python[all]"
cd ..

pip install --upgrade torchaudio

python ./models/download.py --model_name "Qwen/QwQ-32B" #"Qwen/QwQ-32B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B"