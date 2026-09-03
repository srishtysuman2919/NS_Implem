"""Convenience: run every method's built-in self-test."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


MODULES = [
    "methods.rdm_utils",
    "methods.classic_rsa",
    "methods.searchlight_rsa",
    "methods.crossnobis",
    "methods.cka",
    "methods.shape_distance",
    "methods.ridge_encoding",
    "methods.pcm",
    "methods.llm_alignment",
]


def main() -> None:
    for name in MODULES:
        m = importlib.import_module(name)
        st = getattr(m, "_selftest", None)
        if st is None:
            print(f"[skip] {name} (no self-test)")
            continue
        print(f"[run ] {name}")
        try:
            st()
        except AssertionError as e:
            print(f"       FAIL: {e}")
        print()


if __name__ == "__main__":
    main()
