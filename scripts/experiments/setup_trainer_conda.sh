#!/usr/bin/env bash
# Create/refresh the trainer environment as a real conda environment.
#
# Why conda: the previous `work/envs/trainer-py312` was an 11 GB venv inside the
# repo (and was actually Python 3.10). A conda env lives in the conda envs dir,
# is visible in `conda info --envs` / the shell prompt after `conda activate`,
# and can be a genuine Python 3.12 as the runbook requires.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/common_utils/experiment_env.sh"

env_name="${CONDA_ENV_NAME:-risk-girpo}"
trainer_env="${TRAINER_ENV:-/miniconda3/envs/$env_name}"
export PIP_CONSTRAINT=
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$root/work/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" >&2
    exit 2
fi
conda_base="$(conda info --base)"
if [[ -z "${TRAINER_ENV:-}" ]]; then
    trainer_env="$conda_base/envs/$env_name"
fi

if [[ ! -x "$trainer_env/bin/python" ]]; then
    echo "[conda] creating $env_name (python 3.12) at $trainer_env"
    conda create -y -n "$env_name" python=3.12 pip
else
    echo "[conda] reusing $trainer_env"
fi

py="$trainer_env/bin/python"
"$py" -m pip install --upgrade pip setuptools wheel
"$py" -m pip install -r "$root/requirements/mt-agentrisk-trainer.txt"

# veRL hardcodes attn_implementation="flash_attention_2", so flash-attn is
# mandatory. This host has no nvcc; use the prebuilt wheel matching the
# interpreter ABI instead of compiling 2.7.4.post1 from source.
tag="cp$("$py" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
abi="$("$py" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
fa_version="2.8.3"
wheel="$PIP_CACHE_DIR/flash_attn-${fa_version}+cu12torch2.8cxx11abi${abi}-${tag}-${tag}-linux_x86_64.whl"
if [[ ! -s "$wheel" ]]; then
    echo "[conda] downloading $wheel"
    curl -sSL --retry 3 -o "$wheel" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v${fa_version}/$(basename "$wheel")"
fi
"$py" -m pip install --no-deps "$wheel"

# `pip check` reports a cosmetic "decord 0.6.0 is not supported on this
# platform" marker on this image. Fail on any OTHER inconsistency.
pip_check="$("$py" -m pip check 2>&1 || true)"
unexpected="$(printf '%s\n' "$pip_check" | grep -v 'decord .* is not supported on this platform' | grep -v '^$' || true)"
if [[ -n "$unexpected" ]]; then
    printf '%s\n' "$unexpected" >&2
    exit 1
fi
"$py" "$root/scripts/check_trainer_env.py"
echo "[conda] trainer env ready: $trainer_env"
echo "[conda] activate with: conda activate $env_name"
echo "[conda] TRAINER_ENV=$trainer_env"
