"""@file run_all.py
@brief Launch the 2x2 ablation: all four configs as separate processes.

Each subprocess opens its own pygame window and writes logs to a separate run
directory, so the CFL on/off x emergent on/off configurations can be compared
side by side.

Usage:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

CONFIGS = [
    "configs/cfl_on__emergent_on.json",
    "configs/cfl_on__emergent_off.json",
    "configs/cfl_off__emergent_on.json",
    "configs/cfl_off__emergent_off.json",
]

if __name__ == "__main__":
    procs = []
    for cfg in CONFIGS:
        p = subprocess.Popen(
            [sys.executable, "main.py", "--config", cfg],
            cwd=Path(__file__).parent,
        )
        procs.append((cfg, p))
        print(f"[run_all] started PID {p.pid} — {cfg}")

    print(f"\n[run_all] {len(procs)} simulations running. Close each window to stop.\n")

    for cfg, p in procs:
        p.wait()
        print(f"[run_all] PID {p.pid} exited ({cfg})")

    print("[run_all] All done.")
