"""@file run_all.py
@brief Run the 2x2 ablation (CFL on/off x emergent on/off) repeatedly.

Each repetition launches all four configurations concurrently (one pygame window
each), then waits for that batch of four to finish before starting the next
repetition. Every run is written to its own folder:

    logs/<config>/<config>_run<NN>/

so the repeated runs of each configuration are grouped together for averaging.

Usage (from the repo root):
    python tools/run_all.py            # RUNS repetitions (default below)
    python tools/run_all.py 5          # override: 5 repetitions
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

# Number of repetitions per configuration.
RUNS = 10

if __name__ == "__main__":
    runs = RUNS
    if len(sys.argv) > 1:
        try:
            runs = int(sys.argv[1])
        except ValueError:
            print(f"[run_all] ignoring invalid run count {sys.argv[1]!r}; using {RUNS}")

    root = Path(__file__).parent.parent   # repo root (tools/ -> repo root)

    for i in range(1, runs + 1):
        nn = f"{i:02d}"
        print(f"\n[run_all] === repetition {nn}/{runs}: launching "
              f"{len(CONFIGS)} concurrent simulations ===")

        procs = []
        for cfg in CONFIGS:
            name = Path(cfg).stem                 # e.g. cfl_on__emergent_on
            log_dir = f"logs/{name}"
            run_name = f"{name}_run{nn}"
            p = subprocess.Popen(
                [sys.executable, "main.py", "--config", cfg,
                 "--log-dir", log_dir, "--run-name", run_name],
                cwd=root,
            )
            procs.append((name, p))
            print(f"[run_all]   started PID {p.pid} -> logs/{name}/{run_name}")

        # Wait for this batch of four to finish before the next repetition.
        for name, p in procs:
            p.wait()
            print(f"[run_all]   PID {p.pid} done ({name} run {nn})")

    print(f"\n[run_all] All done: {runs} repetition(s) x {len(CONFIGS)} configs.")
