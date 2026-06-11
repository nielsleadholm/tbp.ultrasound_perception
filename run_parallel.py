#!/usr/bin/env python
# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hydra entrypoint for running experiments with episodes executing in parallel.

Run with, e.g.::

    python run_parallel.py experiment=ultrasound_sim2real_sparse_inference

The project's experiment configs are made available via the
``UltrasoundSearchPathPlugin``, so monty's parallel runner can compose them.
"""

from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.run_parallel import main  # noqa: E402

if __name__ == "__main__":
    main()
