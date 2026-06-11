# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Entrypoint for running ultrasound experiments with parallel episodes."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from tbp.monty.experiment.environment import OneObjectPerEpisodeInterface
from tbp.monty.frameworks.run_env import setup_env
from tbp.monty.frameworks.run_parallel import (
    filter_episode_configs,
    generate_parallel_eval_configs,
    generate_parallel_train_configs,
    print_config,
    run_episodes_parallel,
)
from tbp.monty.hydra import register_resolvers

setup_env()


def output_dir_from_run_name(config: DictConfig) -> Path:
    output_dir = (
        Path(config.experiment.config.logging.output_dir)
        / config.experiment.config.logging.run_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


@hydra.main(config_path="conf", config_name="experiment", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    print_config(cfg)
    register_resolvers()

    if cfg.print_cfg:
        return

    cfg.experiment.config.logging.output_dir = str(output_dir_from_run_name(cfg))

    if cfg.experiment.config.do_train:
        train_configs = generate_parallel_train_configs(
            cfg.experiment, cfg.experiment.config.logging.run_name
        )
        train_configs = filter_episode_configs(train_configs, cfg.episodes)
        run_episodes_parallel(
            train_configs,
            cfg.num_parallel,
            cfg.experiment.config.logging.run_name,
            train=True,
        )

    if cfg.experiment.config.do_eval:
        assert issubclass(
            cfg.experiment.config.eval_env_interface_class,
            OneObjectPerEpisodeInterface,
        ), "parallel experiments only work (for now) with per object env interfaces"
        eval_configs = generate_parallel_eval_configs(
            cfg.experiment, cfg.experiment.config.logging.run_name
        )
        eval_configs = filter_episode_configs(eval_configs, cfg.episodes)
        run_episodes_parallel(
            eval_configs,
            cfg.num_parallel,
            cfg.experiment.config.logging.run_name,
            train=False,
        )


if __name__ == "__main__":
    main()
