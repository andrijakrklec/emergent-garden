"""@file main.py
@brief Command-line entry point: parse @c --config and launch the simulation GUI.

Original emergent-garden base by Vishal Paudel (2023); extended for the CFL thesis.
"""

import argparse
from src.game import Game

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    game = Game(config_path=args.config)
    game.run()
    game.quit()