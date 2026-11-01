#!/usr/bin/env python3
from importlib.metadata import version


EXPECTED = {
    "flash-attn": "2.7.4.post1",
    "pyarrow": "19.0.1",
    "transformers": "4.57.3",
    "vllm": "0.11.0",
}


def main() -> None:
    import flash_attn
    import peft
    import ray
    import torch
    import verl

    for package, expected in EXPECTED.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {actual}")
    print(
        "trainer environment: ok "
        f"(torch={torch.__version__}, ray={ray.__version__}, peft={peft.__version__}, "
        f"verl={version('verl')}, flash_attn={flash_attn.__version__})"
    )


if __name__ == "__main__":
    main()
