# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Ultrasound environments compatible with the latest ``tbp.monty`` API.

These environments implement the ``SimulatedEnvironment`` protocol. They return
``Observations`` and ``ProprioceptiveState`` from ``reset`` / ``step``, as expected
by the Monty :class:`~tbp.monty.experiment.environment.Interface`.

The ultrasound "patch" is extracted here (rather than in a dataloader, as in the old
pre-Hydra implementation). The patch, its location in the full image, and the probe
pose are passed to :class:`~custom_classes.sensor_module.UltrasoundSM` for feature
extraction.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import PIL
import quaternion as qt
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import SimulatedEnvironment
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

# The probe is mounted at a fixed offset relative to the tracked agent. The y-component
# (0.028) is the lateral offset and the z-component (0.105) is the distance from the
# tracker origin to the top of the ultrasound image. These are used by the sensor module
# to reconstruct the patch location in world coordinates.
PROBE_SENSOR_OFFSET = np.array([0.0, 0.028, 0.105])

AGENT_ID = "agent_id_0"
SENSOR_ID = "patch"


def find_patch_with_highest_gradient(
    full_image, patch_size, grid_size=9, window_size=100
):
    """Find the first patch with a significant horizontal edge in the ultrasound image.

    Takes a patch at the center top of the image and shifts it down in the middle
    column. Each time it calculates the horizontal edges using a Sobel filter on a
    ``grid_size`` x ``grid_size`` grid of mean intensities on a half-sized patch. Based
    on the center location of the first patch that shows a significant local peak it
    extracts a patch of size ``patch_size`` x ``patch_size``.

    Args:
        full_image: The full ultrasound image of shape (N, M).
        patch_size: Size of the square patch to extract.
        grid_size: Number of bins to group the pixel values into along each dimension
            of the patch for calculating the mean intensity.
        window_size: Size of the window to calculate the local mean and std of the
            gradient.

    Returns:
        tuple: ``(best_patch, y_start)`` where ``best_patch`` is the first patch with a
        significant horizontal edge and ``y_start`` is the y pixel coordinate of the
        start of the patch.
    """
    height, width = full_image.shape
    x_center = width // 2
    start_y = patch_size // 2

    best_location = None
    y_starting_positions = []
    y_central_positions = []
    gradients = []

    # Sobel kernel for horizontal edge detection
    sobel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Collect all gradients by moving a smaller test patch down the center column
    test_patch_size = patch_size // 2
    for y in range(start_y, height - patch_size // 2):
        patch = full_image[
            y - test_patch_size // 2 : y + test_patch_size // 2,
            x_center - test_patch_size // 2 : x_center + test_patch_size // 2,
        ]

        cell_size = test_patch_size // grid_size
        cell_means = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                cell = patch[
                    i * cell_size : (i + 1) * cell_size,
                    j * cell_size : (j + 1) * cell_size,
                ]
                cell_means[i, j] = np.mean(cell)

        padded_means = np.pad(cell_means, 1, mode="edge")

        edge_response = np.zeros_like(cell_means)
        for i in range(grid_size):
            for j in range(grid_size):
                region = padded_means[i : i + 3, j : j + 3]
                edge_response[i, j] = np.sum(region * sobel_horizontal)

        total_gradient = np.sum(np.abs(edge_response))

        # Use the initial y position of the patch for the depth; the location of the
        # edge within the patch is determined later for the final depth reading.
        starting_y = y - patch_size // 2
        y_starting_positions.append(starting_y)
        y_central_positions.append(y)
        gradients.append(total_gradient)

    gradients = np.array(gradients)
    y_starting_positions = np.array(y_starting_positions)
    y_central_positions = np.array(y_central_positions)
    max_gradient = np.max(gradients)

    padded_gradients = np.pad(gradients, window_size, mode="edge")

    # Find the first significant local peak
    for i in range(len(gradients)):
        local_window = padded_gradients[i : i + 2 * window_size]
        local_mean = np.mean(local_window)
        local_std = np.std(local_window)
        local_threshold = local_mean + local_std

        if (
            gradients[i] > gradients[max(0, i - 1)]
            and gradients[i] > gradients[min(len(gradients) - 1, i + 1)]
            and gradients[i] > local_threshold
            and local_std > (np.std(gradients) // 10)
            and gradients[i] > max_gradient / 2
        ):
            best_central_location = (y_central_positions[i], x_center)
            best_starting_location = (y_starting_positions[i], x_center)
            break

    # If no peak found, use the maximum gradient
    if best_location is None:
        max_idx = np.argmax(gradients)
        best_central_location = (y_central_positions[max_idx], x_center)
        best_starting_location = (y_starting_positions[max_idx], x_center)

    y, x = best_central_location
    best_patch = full_image[
        y - patch_size // 2 : y + patch_size // 2,
        x - patch_size // 2 : x + patch_size // 2,
    ]

    y_start, _ = best_starting_location

    return best_patch, y_start


def _to_quaternion(rotation):
    """Convert a (w, x, y, z) sequence or quaternion into a ``qt.quaternion``."""
    if isinstance(rotation, qt.quaternion):
        return rotation
    return qt.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])


class UltrasoundEnvironment(SimulatedEnvironment):
    """Base ultrasound environment.

    Manages a set of "scenes" (one folder of observations per scanned object) and
    extracts the ultrasound patch from each full image. Subclasses define how each raw
    data point (full image + probe pose) is obtained via :meth:`_load_raw_data_point`.

    NOTE: This base class is not fully functional on its own. Use
    :class:`JSONDatasetUltrasoundEnvironment` (offline datasets) or
    :class:`~custom_classes.probe_triggered_environment.ProbeTriggeredUltrasoundEnvironment`
    (live data collection).
    """

    def __init__(self, patch_size: int = 256, data_path: str | None = None):
        """Initialize environment.

        Args:
            patch_size: Height and width of the extracted patch in pixels.
            data_path: Path to the dataset. Each sub-folder is treated as a scene
                (i.e. a scanned object). If ``None``, no scenes are discovered (used by
                streaming subclasses).
        """
        self.patch_size = patch_size
        self.data_path = data_path
        self.full_image = None  # Store the full image for plotting

        if self.data_path is not None and os.path.isdir(self.data_path):
            self.scene_names = sorted(
                a for a in os.listdir(self.data_path) if not a.startswith(".")
            )
        else:
            self.scene_names = []

        # Strip the leading number from scene names, e.g. "002_montys_brain" ->
        # "montys_brain".
        self.object_names = [
            "_".join(name.split("_")[1:]) for name in self.scene_names
        ]

        self.current_scene = 0
        self.step_count = 0

    # ------------------------------------------------------------------
    # SimulatedEnvironment protocol
    # ------------------------------------------------------------------
    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Return the next observation and proprioceptive state.

        Actions are ignored: the environment simply replays the next data point.

        Raises:
            StopIteration: If there is no further data for the current scene.
        """
        obs, state = self._observation_at(self.step_count)
        self.step_count += 1
        return obs, state

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset to the first data point of the current scene."""
        self.step_count = 0
        return self._observation_at(self.step_count)

    def switch_to_scene(self, scene_idx: int) -> None:
        """Load the folder for a new scanned object and reset the step counter."""
        self.current_scene = scene_idx
        self.step_count = 0

    def close(self) -> None:
        self._current_state = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _observation_at(
        self, step: int
    ) -> tuple[Observations, ProprioceptiveState]:
        full_image, state_dict = self._load_raw_data_point(step)
        return self._build_observation_and_state(full_image, state_dict)

    def _load_raw_data_point(self, step: int):
        """Return ``(full_image, state_dict)`` for the given step.

        Must be implemented by subclasses.

        Raises:
            NotImplementedError: Always (subclasses must override).
        """
        raise NotImplementedError

    def _build_observation_and_state(
        self, full_image, state_dict
    ) -> tuple[Observations, ProprioceptiveState]:
        self.full_image = np.asarray(full_image, dtype=float)
        patch, patch_pixel_start = find_patch_with_highest_gradient(
            self.full_image, patch_size=self.patch_size
        )

        observations = Observations(
            {
                AgentID(AGENT_ID): AgentObservations(
                    {
                        SensorID(SENSOR_ID): SensorObservation(
                            {
                                "img": patch,
                                "patch_pixel_start": patch_pixel_start,
                                "full_image_height": self.full_image.shape[0],
                            }
                        ),
                    }
                )
            }
        )

        agent_raw = state_dict[AGENT_ID]
        agent_position = np.array(agent_raw["position"], dtype=float)
        agent_rotation = _to_quaternion(agent_raw["rotation"])
        sensor_rotation = _to_quaternion(
            agent_raw["sensors"]["ultrasound"]["rotation"]
        )

        proprioceptive_state = ProprioceptiveState(
            {
                AgentID(AGENT_ID): AgentState(
                    sensors={
                        SensorID(SENSOR_ID): SensorState(
                            # The probe sensor offset is fixed relative to the tracker.
                            position=PROBE_SENSOR_OFFSET.copy(),
                            rotation=sensor_rotation,
                        ),
                    },
                    position=agent_position,
                    rotation=agent_rotation,
                )
            }
        )
        return observations, proprioceptive_state

    def get_full_image(self):
        """Return the most recently loaded full ultrasound image."""
        if self.full_image is None:
            return np.zeros((256, 256), dtype=np.float32)
        return self.full_image


class JSONDatasetUltrasoundEnvironment(UltrasoundEnvironment):
    """Replays an offline ultrasound dataset stored as JSON files.

    Each scene folder contains ``{step}.json`` files with the keys ``obs`` (the full
    ultrasound image) and ``state`` (the probe pose at collection time).
    """

    def _load_raw_data_point(self, step: int):
        scene = self.scene_names[self.current_scene]
        json_path = Path(self.data_path) / scene / f"{step}.json"
        try:
            with json_path.open("r") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            # Signals the end of the data for this scene.
            raise StopIteration from exc

        full_image = np.array(data["obs"][AGENT_ID]["ultrasound"]["img"])
        return full_image, data["state"]
