# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hydra search-path plugin that exposes this project's ``conf`` tree.

This lets the project's experiment configs (under ``conf/experiment``) be composed
alongside the base building blocks provided by ``tbp.monty`` (via its own
``MontySearchPathPlugin``), without each entrypoint having to hardcode the path.
"""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

# Repo root is three parents up from this file:
# <repo>/hydra_plugins/ultrasound_searchpath_plugin/ultrasound_searchpath_plugin.py
_CONF_DIR = Path(__file__).resolve().parents[2] / "conf"


class UltrasoundSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="tbp.ultrasound_perception",
            path=f"file://{_CONF_DIR}",
        )
