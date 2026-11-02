#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sources="$root/sources"
toolshield="$sources/ToolShield-main"
openhands="$sources/OpenHands-0.54.0"
toolshield_commit="95a2e342492f10e1a61e5c3bcb8df1149005560f"
openhands_commit="4aae68ae53477d05010a509f1f1e3f02e61b1831"

mkdir -p "$sources"

if [[ ! -f "$toolshield/agentrisk/client.py" ]]; then
    git clone --no-checkout https://github.com/CHATS-lab/ToolShield.git "$toolshield"
    git -C "$toolshield" fetch --depth 1 origin "$toolshield_commit"
    git -C "$toolshield" checkout --detach "$toolshield_commit"
fi

actual_toolshield="$(git -C "$toolshield" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$actual_toolshield" && "$actual_toolshield" != "$toolshield_commit" ]]; then
    echo "ToolShield checkout must be $toolshield_commit, got $actual_toolshield" >&2
    exit 2
fi

if [[ ! -f "$openhands/pyproject.toml" ]]; then
    git clone --depth 1 --branch 0.54.0 \
        https://github.com/All-Hands-AI/OpenHands.git "$openhands"
fi

actual_openhands="$(git -C "$openhands" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$actual_openhands" && "$actual_openhands" != "$openhands_commit" ]]; then
    echo "OpenHands checkout must be $openhands_commit, got $actual_openhands" >&2
    exit 2
fi

echo "ToolShield source: $toolshield"
echo "OpenHands source:  $openhands ($openhands_commit)"
