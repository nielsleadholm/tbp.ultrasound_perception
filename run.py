# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Entrypoint for running an ultrasound perception experiment."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.hydra import register_resolvers

setup_env()

logger = logging.getLogger(__name__)


def print_config(config: DictConfig) -> None:
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(OmegaConf.to_yaml(config))
    print("-" * 100)


def output_dir_from_run_name(config: DictConfig) -> Path:
    output_dir = (
        Path(config.experiment.config.logging.output_dir)
        / config.experiment.config.logging.run_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def _configure_probe_triggered_paths(cfg: DictConfig) -> None:
    experiment_name = cfg.experiment.config.logging.run_name
    if "probe_triggered" not in experiment_name:
        return

    object_name = input(
        "Enter the name of the object for this experiment: "
    ).strip()
    if not object_name:
        object_name = "unknown_object"

    object_name = "".join(
        character for character in object_name if character.isalnum() or character in (" ", "-", "_")
    ).rstrip().replace(" ", "_")

    output_dir = (
        Path(cfg.experiment.config.logging.output_dir)
        / experiment_name
        / object_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.experiment.config.logging.output_dir = str(output_dir)
    cfg.experiment.config.environment.env_init_args.save_path = str(
        output_dir / "observations"
    )


@hydra.main(config_path="conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    register_resolvers()
    print_config(cfg)

    if cfg.print_cfg:
        return

    _configure_probe_triggered_paths(cfg)
    if "probe_triggered" not in cfg.experiment.config.logging.run_name:
        cfg.experiment.config.logging.output_dir = str(output_dir_from_run_name(cfg))

    experiment = hydra.utils.instantiate(cfg.experiment)
    start_time = time.time()
    with experiment:
        experiment.run()

    logger.info(
        "Done running %s in %s seconds",
        cfg.experiment.config.logging.run_name,
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
