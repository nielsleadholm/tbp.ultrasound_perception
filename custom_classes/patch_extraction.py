# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Patch extraction utilities for ultrasound images."""

from __future__ import annotations

import numpy as np


def find_patch_with_highest_gradient(
    full_image: np.ndarray,
    patch_size: int,
    grid_size: int = 9,
    window_size: int = 100,
) -> tuple[np.ndarray, int]:
    """Find the first patch with a significant horizontal edge in the ultrasound image.

    Args:
        full_image: The full ultrasound image of shape (N, M).
        patch_size: Size of the square patch to extract.
        grid_size: Number of bins to group pixel values into along each dimension.
        window_size: Size of the window to calculate local mean and std of gradient.

    Returns:
        Tuple of (patch, patch_pixel_start) where patch_pixel_start is the y pixel
        coordinate of the start of the patch.
    """
    height, width = full_image.shape
    x_center = width // 2
    start_y = patch_size // 2

    best_central_location = None
    best_starting_location = None
    y_starting_positions = []
    y_central_positions = []
    gradients = []

    sobel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

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

        starting_y = y - patch_size // 2
        y_starting_positions.append(starting_y)
        y_central_positions.append(y)
        gradients.append(total_gradient)

    gradients = np.array(gradients)
    y_starting_positions = np.array(y_starting_positions)
    y_central_positions = np.array(y_central_positions)
    max_gradient = np.max(gradients)

    padded_gradients = np.pad(gradients, window_size, mode="edge")

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

    if best_central_location is None:
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
