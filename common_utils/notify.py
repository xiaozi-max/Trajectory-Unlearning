#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("status", choices=("start", "success", "failure"))
    parser.add_argument("name")
    parser.add_argument("--detail", default="")
    args = parser.parse_args()
    webhook = os.environ.get("FEISHU_WEBHOOK_URL")
    if not webhook:
        print("[notify] FEISHU_WEBHOOK_URL is unset; skipping")
        return
    payload = {
        "msg_type": "text",
        "content": {"text": f"[{args.status}] {args.name}\n{args.detail}".strip()},
    }
    secret = os.environ.get("FEISHU_WEBHOOK_SECRET", "")
    if secret:
        timestamp = str(int(time.time()))
        key = f"{timestamp}\n{secret}".encode()
        payload["timestamp"] = timestamp
        payload["sign"] = base64.b64encode(hmac.new(key, digestmod=hashlib.sha256).digest()).decode()
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status >= 300:
            raise RuntimeError(f"Feishu returned HTTP {response.status}")


if __name__ == "__main__":
    main()
