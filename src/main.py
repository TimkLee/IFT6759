"""
Reads the configuration file and carries out the instructions to train the desired model.
"""

import argparse
import logging
import os















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Evaluation")

    parser.add_argument(
        "-c",
        "--generate-charts",
        dest="generate_charts",
        help="(boolean) if passed, generate model visuals",
        action="store_true",
    )
    parser.set_defaults(generate_charts=True)

    parser.add_argument(
        "-l",
        "--log-results",
        dest="log_results",
        help="(boolean) if passed, logs model parameters and performance metrics to Comet.ml",
        action="store_true",
    )
    parser.set_defaults(log_results=False)

    args = parser.parse_args()

    main(args)





