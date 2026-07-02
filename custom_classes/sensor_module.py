# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Sensor module for ultrasound data processing.

Turns an ultrasound image patch into a Cortical Messaging Protocol ``Message`` with a
surface normal (pose vectors) and curvature, located in world coordinates using the
probe pose provided by the trackers.
"""

from __future__ import annotations

import numpy as np
import quaternion as qt
from scipy.optimize import least_squares
from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    SensorModule,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import AgentState
from tbp.monty.frameworks.sensors import SensorID


class UltrasoundSM(SensorModule):
    """Sensor module that extracts pose and curvature features from ultrasound patches.

    Implements the latest ``SensorModule`` interface: the proprioceptive state is
    received in :meth:`update_state` and the patch observation in :meth:`step`, which
    returns a CMP :class:`~tbp.monty.cmp.Message`.
    """

    def __init__(self, sensor_module_id: str, save_raw_obs: bool = False):
        super().__init__()
        self.sensor_module_id = sensor_module_id
        self.save_raw_obs = save_raw_obs
        self.is_exploring = False

        # Probe (tracker) pose for the current step, set in update_state.
        self._tracker_position = np.zeros(3)
        self._tracker_orientation = qt.quaternion(1, 0, 0, 0)
        self._probe_position = np.zeros(3)
        self._probe_orientation = qt.quaternion(1, 0, 0, 0)

        self.processed_obs: list = []
        self.raw_observations: list = []
        self.sm_properties: list = []

        self.plotting_data = {
            "column_points": [],
            "center_edge": None,
            "fitted_circle": None,
            "point_normal": None,
            "curvature": None,
            "mean_depth": 0.0,
            "observed_locations": [],
            "normal_rel_world": [],
        }

    # ------------------------------------------------------------------
    # SensorModule interface
    # ------------------------------------------------------------------
    def pre_episode(self) -> None:
        self.is_exploring = False
        self.processed_obs = []
        self.raw_observations = []
        self.sm_properties = []
        self.plotting_data["observed_locations"] = []
        self.plotting_data["normal_rel_world"] = []

    def state_dict(self):
        return {
            "raw_observations": self.raw_observations,
            "sm_properties": self.sm_properties,
            "processed_observations": self.processed_obs,
        }

    def update_state(self, agent: AgentState):
        """Store the probe/tracker pose for the current step."""
        sensor = agent.sensors[SensorID(self.sensor_module_id)]
        self._tracker_position = np.array(agent.position)
        self._tracker_orientation = self._as_quaternion(agent.rotation)
        self._probe_position = np.array(sensor.position)
        self._probe_orientation = self._as_quaternion(sensor.rotation)

    def step(
        self,
        ctx: RuntimeContext,  # noqa: ARG002
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message:
        if self.save_raw_obs and not self.is_exploring:
            self.raw_observations.append(observation)
            self.sm_properties.append(
                dict(
                    sm_rotation=qt.as_float_array(self._tracker_orientation),
                    sm_location=np.array(self._tracker_position),
                )
            )

        tracker_position = self._tracker_position
        probe_position = self._probe_position
        tracker_orientation = self._tracker_orientation

        # normal_rel_patch points up (in the y direction) in the image plane. In the
        # world frame it should point towards the agent.
        normal_rel_patch, curvature, pixel_depth_in_patch = (
            self.extract_patch_pose_feat(observation["img"])
        )

        # Derive depth from pixel location in image (in meters)
        pixel_depth_in_image = observation["patch_pixel_start"] + pixel_depth_in_patch
        patch_depth = self.get_depth_from_pixel_location(
            observation["full_image_height"], pixel_depth_in_image
        )

        patch_world_location = self.get_patch_world_location(
            tracker_position,
            probe_position,
            tracker_orientation,
            patch_depth,
        )

        self.plotting_data["observed_locations"].append(patch_world_location)

        # To ensure normal_rel_world points towards the agent, we negate
        # normal_rel_patch before applying the tracker orientation.
        normal_rel_world = qt.as_rotation_matrix(tracker_orientation) @ (
            -normal_rel_patch
        )
        self.plotting_data["normal_rel_world"].append(normal_rel_world)
        patch_world_orientation = self.calculate_patch_world_orientation(
            normal_rel_world
        )

        # Store data for plotting
        self.plotting_data["mean_depth"] = patch_depth
        self.plotting_data["point_normal"] = normal_rel_patch
        self.plotting_data["curvature"] = curvature

        percept = Message(
            location=patch_world_location,
            morphological_features={
                "pose_vectors": patch_world_orientation,
                "pose_fully_defined": False,
            },
            non_morphological_features={
                # NOTE: This will not match with any of the curvature features in
                # pretrained models from the simulation.
                "curvature": curvature,
            },
            confidence=1,
            use_state=not motor_only_step,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

        if not self.is_exploring:
            self.processed_obs.append(percept.__dict__)

        return percept

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_quaternion(rotation):
        if isinstance(rotation, qt.quaternion):
            return rotation
        return qt.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])

    def get_depth_from_pixel_location(
        self, full_image_height, pixel_depth_in_image, max_depth=7
    ):
        """Crop image and calculate depth as a percentage from the top.

        Args:
            full_image_height: The height of the full ultrasound image in pixels.
            pixel_depth_in_image: The y pixel coordinate of the depth reading within the
                image.
            max_depth: Depth setting of the ultrasound probe, in cm.

        Returns:
            Estimated depth of patch in meters.
        """
        depth_perc = pixel_depth_in_image / full_image_height
        depth_cm = depth_perc * max_depth
        depth_m = depth_cm / 100

        return depth_m

    def get_patch_world_location(
        self,
        agent_position,
        sensor_position,
        agent_rotation,
        image_depth,
    ):
        """Calculate the patch's location in world coordinates."""
        offset_direction = np.array([0.0, 0.0, 1.0])
        rotated_offset_direction = (
            qt.as_rotation_matrix(agent_rotation) @ offset_direction
        )
        offset_distance = sensor_position[1]  # = 0.028
        relative_offset = offset_distance * rotated_offset_direction
        agent_position = agent_position + relative_offset

        fake_sensor_rel_world = qt.as_rotation_matrix(agent_rotation) @ np.array(
            [0.0, 1.0, 0.0]
        )
        offset_distance = sensor_position[2] + image_depth

        # The movement vector is in the opposite direction of the unit normal, scaled by
        # the offset distance.
        movement_vector = -fake_sensor_rel_world * offset_distance

        patch_world_location = agent_position + movement_vector

        return patch_world_location

    def calculate_patch_world_orientation(self, patch_normal_in_world_frame):
        """Calculate the patch's orientation in the world frame.

        Args:
            patch_normal_in_world_frame: The patch's surface normal vector, already
                expressed in world coordinates.

        Returns:
            A 3x3 matrix whose rows are orthonormal basis vectors (normal, dir1, dir2)
            expressed in world coordinates.
        """
        normal_vector = patch_normal_in_world_frame / np.linalg.norm(
            patch_normal_in_world_frame
        )

        world_z_axis = np.array([0.0, 0.0, 1.0])
        dir1 = np.cross(world_z_axis, normal_vector)

        if np.linalg.norm(dir1) < 1e-6:
            dir1 = np.array([1.0, 0.0, 0.0])

        dir1 /= np.linalg.norm(dir1)

        dir2 = np.cross(normal_vector, dir1)
        dir2 /= np.linalg.norm(dir2)

        patch_orientation_in_world = np.vstack([normal_vector, dir1, dir2])

        return patch_orientation_in_world

    def extract_patch_pose_feat(self, patch):
        """Extract surface normal, curvature and edge depth from a grayscale patch.

        First extracts edge points, then fits a circle to these points, and finally
        calculates the normal vector based on the fitted curve at the center location.

        Args:
            patch: Grayscale image containing a white edge on a black background.

        Returns:
            tuple: ``(normal_3d, curvature, y_depth_in_image)``.
        """
        all_column_points_tuples, center_edge_point = self.extract_edge_points(patch)
        all_column_points_np = np.array(all_column_points_tuples)

        self.plotting_data["column_points"] = all_column_points_tuples
        self.plotting_data["center_edge"] = center_edge_point

        fit_params, curvature = self.fit_circle_to_points(
            patch, all_column_points_np, center_edge_point
        )

        if fit_params is not None and not fit_params.get("is_line", True):
            self.plotting_data["fitted_circle"] = (
                fit_params["center"][0],
                fit_params["center"][1],
                fit_params["radius"],
            )
        else:
            self.plotting_data["fitted_circle"] = None

        normal_3d = self.calculate_normal_from_fit(fit_params, center_edge_point)

        y_depth_in_image = center_edge_point[1]

        return normal_3d, curvature, y_depth_in_image

    def extract_edge_points(self, image):
        """Extract edge points by scanning each column for above-threshold intensities.

        Args:
            image: Grayscale image.

        Returns:
            tuple: ``(all_column_points_tuples, center_edge_point)``.
        """
        image_height, image_width = image.shape[:2]

        max_intensity = np.max(image)
        WHITE_THRESHOLD = max(50, 0.3 * max_intensity)

        Y_SCAN_RANGE = 50
        center_x = image_width // 2
        y_center = image_height // 2

        for y_row in range(image_height):
            if image[y_row, center_x] > WHITE_THRESHOLD:
                y_center = y_row
                break

        y_min_scan = max(0, y_center - Y_SCAN_RANGE)
        y_max_scan = min(image_height, y_center + Y_SCAN_RANGE + 1)

        all_column_points_tuples = []
        center_edge_point = None

        for x_col in range(image_width):
            for y_row in range(y_min_scan, y_max_scan):
                if image[y_row, x_col] > WHITE_THRESHOLD:
                    point = (float(x_col), float(y_row))
                    all_column_points_tuples.append(point)

                    if x_col == center_x:
                        center_edge_point = point
                    break

        if center_edge_point is None:
            center_edge_point = (float(center_x), float(y_center))

        return all_column_points_tuples, center_edge_point

    def fit_circle_to_points(self, image, all_column_points_np, edge_point):
        """Fit a circle to the detected edge points to get curvature.

        Also makes sure the circle passes through the edge point at the center column.

        Args:
            image: The input image (unused, kept for parity).
            all_column_points_np: Array of (x,y) points representing the edge.
            edge_point: (x,y) coordinates of the center edge point.

        Returns:
            tuple: ``(fit_params, curvature)``.
        """
        if not isinstance(all_column_points_np, np.ndarray):
            all_column_points_np = np.array(all_column_points_np)

        if len(all_column_points_np) < 3:
            fit_params = {
                "is_line": True,
                "points_used": [edge_point],
                "retry_count": 0,
            }
            final_curvature = 0.0
            return fit_params, final_curvature

        def fit_circle_through_point(points, P0, calc_error=False):
            pts = np.asarray(points, dtype=float)
            x0, y0 = P0

            def residuals(ab):
                a, b = ab
                r = np.hypot(x0 - a, y0 - b)
                return np.hypot(pts[:, 0] - a, pts[:, 1] - b) - r

            x, y = pts[:, 0], pts[:, 1]
            A = np.c_[2 * x, 2 * y, np.ones_like(x)]
            try:
                c, _, _, _ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
                a0, b0 = c[0], c[1]
            except np.linalg.LinAlgError:
                a0, b0 = np.mean(pts[:, 0]), np.mean(pts[:, 1])

            try:
                res = least_squares(residuals, x0=[a0, b0], method="trf")
                a, b = res.x
                r = np.hypot(x0 - a, y0 - b)

                if calc_error:
                    final_residuals = residuals([a, b])
                    mse = np.mean(np.square(final_residuals))
                    return (a, b), r, mse
                else:
                    return (a, b), r
            except Exception:
                if calc_error:
                    return None, float("inf"), float("inf")
                else:
                    return None, float("inf")

        def normalize_and_fit(points, P0):
            mean_x = np.mean(points[:, 0])
            mean_y = np.mean(points[:, 1])
            std_xy = np.std(points, axis=0).mean()

            if std_xy < 1e-8:
                std_xy = 1.0

            norm_points = (points - [mean_x, mean_y]) / std_xy
            norm_P0 = ((P0[0] - mean_x) / std_xy, (P0[1] - mean_y) / std_xy)

            center_norm, radius_norm, fit_error = fit_circle_through_point(
                norm_points, norm_P0, calc_error=True
            )

            return center_norm, radius_norm, fit_error, mean_x, mean_y, std_xy

        center_norm, radius_norm, fit_error, mean_x, mean_y, std_xy = normalize_and_fit(
            all_column_points_np, edge_point
        )

        use_subset = False
        best_subset_error = float("inf")
        best_subset_params = None
        best_subset_points = None
        attempt_num_for_best_subset = 0
        retry_count = 0

        subset_percentages = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        min_fit_error = 0.05

        for percentage in subset_percentages:
            retry_count += 1

            total_points = len(all_column_points_np)
            points_to_keep = int(total_points * percentage)

            if points_to_keep < 10:
                points_to_keep = min(10, total_points)

            points_to_remove = (total_points - points_to_keep) // 2
            if 2 * points_to_remove + points_to_keep > total_points:
                points_to_remove = (total_points - points_to_keep) // 2

            start_idx = points_to_remove
            end_idx = total_points - points_to_remove

            if start_idx < 0:
                start_idx = 0
            if end_idx > total_points:
                end_idx = total_points
            if end_idx - start_idx < 3:
                continue

            subset_points = all_column_points_np[start_idx:end_idx]

            try:
                (
                    center_norm_subset,
                    radius_norm_subset,
                    fit_error_subset,
                    mean_x_subset,
                    mean_y_subset,
                    std_xy_subset,
                ) = normalize_and_fit(subset_points, edge_point)

                if radius_norm_subset <= 0 or radius_norm_subset > 1e6:
                    continue

                if fit_error_subset < best_subset_error:
                    best_subset_error = fit_error_subset
                    best_subset_params = (
                        center_norm_subset,
                        radius_norm_subset,
                        fit_error_subset,
                        mean_x_subset,
                        mean_y_subset,
                        std_xy_subset,
                    )
                    best_subset_points = subset_points
                    attempt_num_for_best_subset = retry_count

            except Exception as e:
                print(f"  Error in retry {retry_count}: {str(e)}")

        if best_subset_params is not None:
            if fit_error > min_fit_error or radius_norm <= 0 or radius_norm > 1e6:
                use_subset = True

            if use_subset:
                center_norm, radius_norm, fit_error, mean_x, mean_y, std_xy = (
                    best_subset_params
                )
                points_to_use = best_subset_points
            else:
                points_to_use = all_column_points_np
        else:
            points_to_use = all_column_points_np

        final_displayed_retry_count = 0
        if use_subset:
            final_displayed_retry_count = attempt_num_for_best_subset

        if center_norm is None or radius_norm > 1e6 or radius_norm < 1e-6:
            fit_params = {
                "is_line": True,
                "points_used": [edge_point],
                "retry_count": final_displayed_retry_count,
            }
            final_curvature = 0.0
        else:
            center_x = center_norm[0] * std_xy + mean_x
            center_y = center_norm[1] * std_xy + mean_y
            radius = radius_norm * std_xy

            final_curvature = 1.0 / radius if radius > 1e-6 else 0.0

            points_tuples = [tuple(pt) for pt in points_to_use]

            fit_params = {
                "is_line": False,
                "center": (center_x, center_y),
                "radius": radius,
                "points_used": [edge_point] + points_tuples,
                "inliers_percent": len(points_to_use) / len(all_column_points_np)
                if len(all_column_points_np) > 0
                else 0.0,
                "retry_count": final_displayed_retry_count,
                "fit_error": fit_error,
                "used_subset": use_subset,
            }

        return fit_params, final_curvature

    def calculate_normal_from_fit(self, fit_params, edge_point):
        """Calculate the normal vector based on the fitted circle/line.

        Args:
            fit_params: Dictionary containing fit parameters.
            edge_point: (x,y) of the point where we want to calculate the normal.

        Returns:
            3D normal vector [nx, ny, 0].
        """
        x0, y0 = edge_point

        if fit_params.get("is_line", True):
            normal = np.array([0.0, -1.0, 0.0])
        else:
            circle_center = fit_params["center"]
            cx, cy = circle_center

            dx = x0 - cx
            dy = y0 - cy

            magnitude = np.hypot(dx, dy)
            if magnitude > 1e-9:
                nx = dx / magnitude
                ny = dy / magnitude
            else:
                nx, ny = 0.0, -1.0

            if ny > 0:
                nx = -nx
                ny = -ny

            normal = np.array([nx, ny, 0.0])

        return normal
