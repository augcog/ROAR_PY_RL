from typing import List, Optional
from roar_py_interface import RoarPyActor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor, RoarPyCustomLambdaSensor, RoarPyCustomLambdaSensorData
from .base_env import RoarRLEnv
from typing import Any, Dict, SupportsFloat, Tuple, Optional, Set
import gymnasium as gym
import numpy as np
from shapely import Polygon, Point
from collections import OrderedDict

def distance_to_waypoint_polygon(
    waypoint_1: RoarPyWaypoint,
    waypoint_2: RoarPyWaypoint,
    point: np.ndarray
):
    p1, p2 = waypoint_1.line_representation
    p3, p4 = waypoint_2.line_representation
    polygon = Polygon([p1, p2, p4, p3])
    return polygon.distance(Point(point))

class RoarRLSimEnv(RoarRLEnv):
    def __init__(
            self,
            actor: RoarPyActor,
            manuverable_waypoints: List[RoarPyWaypoint],
            location_sensor : RoarPyLocationInWorldSensor,
            velocimeter_sensor : RoarPyVelocimeterSensor,
            collision_sensor : RoarPyCollisionSensor,
            collision_threshold : float = 30.0,
            waypoint_information_distances : Set[float] = set([-8.0, -5.0, -2.0, 1.0, 2.0, 5.0, 8.0, 10.0, 20.0]),
            world: Optional[RoarPyWorld] = None,
            render_mode="rgb_array"
        ) -> None:
        super().__init__(actor, manuverable_waypoints, world, render_mode)
        self.location_sensor = location_sensor
        self.velocimeter_sensor = velocimeter_sensor
        self.collision_sensor = collision_sensor
        self.collision_threshold = collision_threshold
        self.waypoint_information_distances = waypoint_information_distances
        self._current_waypoint_idx = 0
        self._travelled_dist = 0.0
        self._delta_distance_travelled = 0.0

        self._dists_between_waypoints : List[float] = []
        for i in range(len(self.manuverable_waypoints)):
            waypoint = self.manuverable_waypoints[i]
            next_waypoint = self.manuverable_waypoints[(i+1)%len(self.manuverable_waypoints)]
            self._dists_between_waypoints.append(np.linalg.norm(waypoint.location - next_waypoint.location))
        self._total_dist = np.sum(self._dists_between_waypoints)

    @property
    def observation_space(self) -> gym.Space:
        space = super().observation_space
        if len(self.waypoint_information_distances) > 0:
            waypoints_info_space_dict = OrderedDict()
            for dist in sorted(self.waypoint_information_distances):
                waypoints_info_space_dict[f"waypoint_{dist}"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4,), # x, y, yaw, lane_width
                    dtype=np.float32
                )
            space["waypoints_information"] = gym.spaces.Dict(waypoints_info_space_dict)
        
        return space

    def observation(self) -> Dict[str, Any]:
        obs = super().observation()
        if len(self.waypoint_information_distances) > 0:
            waypoint_info = {}
            dists_negative = sorted([dist for dist in self.waypoint_information_distances if dist < 0], reverse=True)
            dists_positive = sorted([dist for dist in self.waypoint_information_distances if dist >= 0])
            if len(dists_negative) > 0:
                traced_dist = 0.0
                traced_waypoint_begin_idx = self._current_waypoint_idx
                traced_negative_dist_idx = 0
                i_trace = 0
                while traced_negative_dist_idx < len(dists_negative):
                    traced_waypoint_begin_idx = (i_trace + self.manuverable_waypoints) % len(self.manuverable_waypoints)
                    traced_waypoint_end_idx = (traced_waypoint_begin_idx + 1) % len(self.manuverable_waypoints)
                    traced_waypoint_begin = self.manuverable_waypoints[traced_waypoint_begin_idx]
                    traced_waypoint_end = self.manuverable_waypoints[traced_waypoint_end_idx]
                    current_dist_segment = self._dists_between_waypoints[traced_waypoint_begin_idx] if i_trace != 0 else self._travelled_dist - np.sum(self._dists_between_waypoints[:self._current_waypoint_idx])
                    if i_trace == 0:
                        traced_waypoint_end = RoarPyWaypoint.interpolate(
                            traced_waypoint_begin,
                            traced_waypoint_end,
                            current_dist_segment / self._dists_between_waypoints[traced_waypoint_begin_idx]
                        )

                    while traced_dist + current_dist_segment >= -dists_negative[traced_negative_dist_idx]:
                        current_negative_dist = dists_negative[traced_negative_dist_idx]
                        segment_reverse_portion = -current_negative_dist - traced_dist
                        segment_reverse_portion /= current_dist_segment
                        segment_reverse_portion = np.clip(segment_reverse_portion, 0.0, 1.0)
                        
                        interpolated_waypoint = RoarPyWaypoint.interpolate(
                            traced_waypoint_end,
                            traced_waypoint_begin,
                            segment_reverse_portion
                        )

                        waypoint_info[f"waypoint_{current_negative_dist}"] = np.concatenate([
                            interpolated_waypoint.location[:2], interpolated_waypoint.roll_pitch_yaw[2:3], [interpolated_waypoint.lane_width]
                        ])
                        traced_negative_dist_idx += 1
                        if traced_negative_dist_idx >= len(dists_negative):
                            break

                    traced_dist += current_dist_segment
                    i_trace += 1
            if len(dists_positive) > 0:
                traced_dist = 0.0
                traced_waypoint_begin_idx = self._current_waypoint_idx
                traced_positive_dist_idx = 0
                i_trace = 0
                while traced_positive_dist_idx < len(dists_positive):
                    traced_waypoint_begin_idx = (i_trace + self.manuverable_waypoints) % len(self.manuverable_waypoints)
                    traced_waypoint_end_idx = (traced_waypoint_begin_idx + 1) % len(self.manuverable_waypoints)
                    traced_waypoint_begin = self.manuverable_waypoints[traced_waypoint_begin_idx]
                    traced_waypoint_end = self.manuverable_waypoints[traced_waypoint_end_idx]
                    current_dist_segment = self._dists_between_waypoints[traced_waypoint_begin_idx] if i_trace != 0 else np.sum(self._dists_between_waypoints[:self._current_waypoint_idx + 1]) - self._travelled_dist
                    if i_trace == 0:
                        traced_waypoint_begin = RoarPyWaypoint.interpolate(
                            traced_waypoint_end,
                            traced_waypoint_begin,
                            current_dist_segment / self._dists_between_waypoints[traced_waypoint_begin_idx]
                        )

                    while traced_dist + current_dist_segment >= dists_positive[traced_positive_dist_idx]:
                        current_positive_dist = dists_positive[traced_positive_dist_idx]
                        segment_portion = current_positive_dist - traced_dist
                        segment_portion /= current_dist_segment
                        segment_portion = np.clip(segment_portion, 0.0, 1.0)
                        
                        interpolated_waypoint = RoarPyWaypoint.interpolate(
                            traced_waypoint_begin,
                            traced_waypoint_end,
                            segment_portion
                        )

                        waypoint_info[f"waypoint_{current_positive_dist}"] = np.concatenate([
                            interpolated_waypoint.location[:2], interpolated_waypoint.roll_pitch_yaw[2:3], [interpolated_waypoint.lane_width]
                        ])
                        traced_positive_dist_idx += 1
                        if traced_positive_dist_idx >= len(dists_positive):
                            break

                    traced_dist += current_dist_segment
                    i_trace += 1
            obs["waypoints_information"] = waypoint_info
        return obs

    
    def search_waypoint(self, location : np.ndarray) -> int:
        smallest_waypoint_dist = float("inf")
        smallest_waypoint_idx = 0
        for i in range(self.current_waypoint_idx - 10, self.current_waypoint_idx - 10 + len(self.manuverable_waypoints)):
            waypoint_idx = i % len(self.manuverable_waypoints)
            next_waypoint_idx = (waypoint_idx + 1) % len(self.manuverable_waypoints)
            waypoint = self.manuverable_waypoints[waypoint_idx]
            next_waypoint = self.manuverable_waypoints[next_waypoint_idx]
            dist = distance_to_waypoint_polygon(waypoint, next_waypoint, location)
            if dist == 0.0:
                return waypoint_idx
            if dist < smallest_waypoint_dist:
                smallest_waypoint_dist = dist
                smallest_waypoint_idx = waypoint_idx
        return smallest_waypoint_idx

    def update_travelled_dist(self, location : np.ndarray) -> float:
        new_waypoint_idx = self.search_waypoint(location)
        new_dist = np.sum(self._dists_between_waypoints[:new_waypoint_idx])
        new_start_waypoint, new_end_waypoint = self.manuverable_waypoints[new_waypoint_idx], self.manuverable_waypoints[(new_waypoint_idx+1)%len(self.manuverable_waypoints)]
        
        if np.linalg.norm(new_start_waypoint.location - location) > 1e-6:
            delta_new_segment = (new_end_waypoint.location - new_start_waypoint.location)
            length_new_segment = np.linalg.norm(delta_new_segment)
            unit_delta_segment = delta_new_segment / length_new_segment
            new_dist += np.clip(np.inner(location - new_start_waypoint.location, unit_delta_segment), 0.0, length_new_segment)
        
        delta_dist = self._total_dist + new_dist - self._travelled_dist
        delta_dist %= self._total_dist
        self._travelled_dist = new_dist
        self._current_waypoint_idx = new_waypoint_idx
        return delta_dist

    def reset_vehicle(self) -> None:
        return NotImplementedError

    @property
    def sensors_to_update(self) -> List[Any]:
        return [self.location_sensor, self.velocimeter_sensor, self.collision_sensor]

    def get_reward(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> SupportsFloat:
        collision_impulse = self.collision_sensor.get_last_observation().impulse_normal
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm >= self.collision_threshold:
            penalty = collision_impulse_norm - self.collision_threshold
            penalty = (penalty ** 1.4) / self.collision_threshold
            return -penalty

        normalized_rew = self._delta_distance_travelled
        if normalized_rew < 0:
            return np.exp(normalized_rew) # Gaussian-like penalty for going backwards
        else:
            return normalized_rew + 1
    
    def _step(self, action: Any) -> None:
        location = self.location_sensor.get_last_gym_observation()
        self._delta_distance_travelled = self.update_travelled_dist(location)

    def _reset(self) -> None:
        location = self.location_sensor.get_last_gym_observation()
        self.update_travelled_dist(location)
        self._delta_distance_travelled = 0.0

    def is_terminated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        collision_impulse = self.collision_sensor.get_last_observation().impulse_normal
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm >= self.collision_threshold:
            return True
        return False

    def is_truncated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        return False
