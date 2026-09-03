"""Prepare every dataset used by the experiments.

Run this once before the notebooks / scripts under experiments/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))


def main() -> None:
    import prepare_kriegeskorte92
    import prepare_haxby
    import prepare_nsd
    import prepare_pereira2018
    import prepare_algonauts_tutorial

    print("== Kriegeskorte 92 ==")
    prepare_kriegeskorte92.main()
    print()
    print("== Haxby 2001 (agreement and searchlight) ==")
    prepare_haxby.main()
    print()
    print("== NSD model-brain data ==")
    prepare_nsd.main()
    print()
    print("== Pereira 2018 (LLM alignment) ==")
    prepare_pereira2018.main()
    print()
    print("== Algonauts tutorial (optional) ==")
    prepare_algonauts_tutorial.main()
    print()
    print("all datasets ready under data/cache/")


if __name__ == "__main__":
    main()
