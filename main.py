"""@file main.py
@brief Command-line entry point: parse @c --config and launch the simulation GUI.

Original emergent-garden base by Vishal Paudel (2023); extended for the CFL thesis.
"""

import argparse
from src.game import Game

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--log-dir", default="logs", help="Base directory for log output")
    parser.add_argument("--run-name", default=None,
                        help="Explicit run-folder name (overrides the timestamped default); for batch runs")
    parser.add_argument("--preset", default=None,
                        help="Override emergent_preset from config.json (name of a file in configs/emergent/)")
    args = parser.parse_args()

    game = Game(config_path=args.config, log_dir=args.log_dir, run_name=args.run_name, preset=args.preset)
    game.run()
    game.quit()