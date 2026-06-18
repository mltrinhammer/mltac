"""Compatibility wrapper for the CC MoE 1 expert launcher."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_moe1_experts_4gpu import main  # noqa: E402


if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv[1:1] = ["--domain", "CC"]
    main()
