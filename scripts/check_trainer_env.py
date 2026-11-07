#!/usr/bin/env python3
from importlib.metadata import version


EXPECTED = {
    "pyarrow": "19.0.1",
    "transformers": "4.57.3",
    "vllm": "0.11.0",
}

# flash-attn 2.7.4.post1 must be compiled from source (needs nvcc). Hosts
# without a CUDA toolchain use the ABI-matched prebuilt 2.8.3 wheel instead;
# both provide the flash_attention_2 backend that veRL hardcodes.
FLASH_ATTN_ALLOWED = {"2.7.4.post1", "2.8.3"}


def main() -> None:
    import sys

    import flash_attn
    import peft
    import ray
    import torch
    import verl

    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python != "3.12":
        raise RuntimeError(f"python: expected 3.12, found {python}")

    for package, expected in EXPECTED.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {actual}")
    if flash_attn.__version__ not in FLASH_ATTN_ALLOWED:
        raise RuntimeError(
            f"flash-attn: expected one of {sorted(FLASH_ATTN_ALLOWED)}, "
            f"found {flash_attn.__version__}"
        )
    print(
        "trainer environment: ok "
        f"(python={python}, torch={torch.__version__}, ray={ray.__version__}, "
        f"peft={peft.__version__}, verl={version('verl')}, "
        f"flash_attn={flash_attn.__version__})"
    )


if __name__ == "__main__":
    main()
