"""Download only terminal + filesystem harmful and benign MT-AgentRisk tasks.

`hf download --local-dir --include` triggers huggingface_hub's bulk safe-delete
confirmation (it wants to prune files outside the include set) and aborts
mid-download. Downloading per file avoids that entirely.

Usage:
    HF_TOKEN=... python scripts/download_mt_subset.py \
        --output /ABS/PATH/TO/risk-aware-girpo/work/datasets/MT-AgentRisk
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "CHATS-Lab/MT-AgentRisk"
PREFIXES = (
    "workspaces/terminal/",
    "workspaces/filesystem/",
    "workspaces/benign_tasks/benign_tasks_terminal/",
    "workspaces/benign_tasks/benign_tasks_filesystem/",
)
EXTRA = {"multi_dataset.csv", "single_dataset.csv", "README.md"}


def wanted(path: str) -> bool:
    return path in EXTRA or any(path.startswith(prefix) for prefix in PREFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "work/datasets/MT-AgentRisk",
        help="local dataset root (default: <repo>/work/datasets/MT-AgentRisk)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required (gated dataset)")

    root: Path = args.output
    root.mkdir(parents=True, exist_ok=True)

    def fetch(path: str) -> str:
        target = root / path
        if target.exists() and target.stat().st_size > 0:
            return "skip"
        hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=path,
            local_dir=str(root),
            token=os.environ.get("HF_TOKEN"),
        )
        return "ok"

    files = [
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset")
        if wanted(f)
    ]
    print(f"[download] {len(files)} matching files -> {root}", flush=True)
    done = {"skip": 0, "ok": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, path): path for path in files}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                done[future.result()] += 1
            except Exception as exc:  # noqa: BLE001
                done["error"] += 1
                print(
                    f"[download] FAIL {futures[future]}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if i % 200 == 0:
                print(f"[download] {i}/{len(files)} {done}", flush=True)
    print(f"[download] final {done}", flush=True)
    if done["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
