#!/usr/bin/env python
# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hydra entrypoint for running ultrasound perception experiments.

Run an experiment with, e.g.::

    python run.py experiment=ultrasound_sim2real_sparse_inference

Experiment configs live under ``conf/experiment``. Base building blocks (motor
systems, learning modules, etc.) are composed from both this project's ``conf`` tree
and ``tbp.monty``'s ``conf`` tree (made available via the ``MontySearchPathPlugin``).
"""

from tbp.monty.frameworks.run_env import setup_env

setup_env()

import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tbp.monty.hydra import register_resolvers  # noqa: E402

logger = logging.getLogger(__name__)


def print_config(config: DictConfig) -> None:
    """Print config with nice formatting."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(OmegaConf.to_yaml(config))
    print("-" * 100)


def output_dir_from_run_name(config: DictConfig) -> Path:
    """Build the run-specific output directory (output_dir / run_name).

    For probe-triggered data collection, prompt for the object name and nest the
    output (and the dataset save path) under it, mirroring the old semi-manual
    collection pipeline.

    Returns:
        Path to the run-specific output directory.
    """
    logging_config = config.experiment.config.logging
    output_dir = Path(logging_config.output_dir) / logging_config.run_name

    if "probe_triggered" in str(logging_config.run_name):
        prompt = "Enter the name of the object for this experiment: "
        object_name = input(prompt).strip()
        if not object_name:
            object_name = "unknown_object"
        object_name = "".join(
            c for c in object_name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        object_name = object_name.replace(" ", "_")
        output_dir = output_dir / object_name
        config.experiment.config.environment.env_init_args.save_path = str(
            output_dir / "observations"
        )

    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


@hydra.main(config_path="conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig):
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    register_resolvers()
    print_config(cfg)

    cfg.experiment.config.logging.output_dir = str(output_dir_from_run_name(cfg))

    if cfg.print_cfg:
        return

    experiment = hydra.utils.instantiate(cfg.experiment)
    start_time = time.time()
    with experiment:
        experiment.run()

    logger.info(f"Done running {experiment} in {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
