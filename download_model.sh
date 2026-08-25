#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model"
mkdir -p "$MODEL_DIR"

fetch() {
  local file="$1" url="$2"
  if [ -f "$file" ] && [ -s "$file" ]; then
    echo "Already present: $file -- skipping."
  else
    echo "Downloading $file ..."
    if command -v wget &>/dev/null; then
      wget -c --progress=bar:force -O "$file" "$url"
    else
      curl -L -C - --progress-bar -o "$file" "$url"
    fi
  fi
  head -c1 "$file" | grep -qE '^[<{]' \
    && { echo "ERROR: $file looks like an error page (HTML/JSON), not a GGUF file. Check auth requirements on the source URL."; exit 1; } \
    || echo "OK: $file looks like a valid binary."
}

# Primary and only model: French clinical document generation.
# Q4_K_M was tested and appeared to throttle CPU on an earlier (RAM-constrained)
# machine; staying on Q2_K, which was verified thermally safe (75C peak,
# not throttled) and well under the 7GB RAM ceiling -- see REPORT.md Benchmarks.
fetch "$MODEL_DIR/Ministral-3-3B-Instruct-2512-Q2_K.gguf" \
  "https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q2_K.gguf?download=true"

