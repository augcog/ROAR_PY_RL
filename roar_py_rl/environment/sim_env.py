from typing import List, Optional
from roar_py_interface import RoarPyActor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor
from .base_env import RoarRLEnv
from typing import Any, Dict, SupportsFloat, Tuple, Optional
import gymnasium as gym
import numpy as np

class RoarRLSimEnv(RoarRLEnv):
    def __init__(
            self, 
            actor: RoarPyActor, 
            manuverable_waypoints: List[RoarPyWaypoint], 
            location_sensor : RoarPyLocationInWorldSensor, 
            velocimeter_sensor : RoarPyVelocimeterSensor,
            collision_sensor : RoarPyCollisionSensor, 
            collision_threshold : float = 30.0,
            lookahead_distance : float = 3.0,
            world: Optional[RoarPyWorld] = None, 
            render_mode="rgb_array"
        ) -> None:
        assert location_sensor in self
        super().__init__(actor, manuverable_waypoints, world, render_mode)
        self.location_sensor = location_sensor
        self.velocimeter_sensor = velocimeter_sensor
        self.collision_sensor = collision_sensor
        self.collision_threshold = collision_threshold
        self.lookahead_distance = lookahead_distance
    
    def get_reward(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> SupportsFloat:
        collision_impulse = self.collision_sensor.get_last_observation().impulse_normal
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm >= self.collision_threshold:
            penalty = collision_impulse_norm - self.collision_threshold
            penalty = (penalty ** 1.4) / self.collision_threshold
            return -penalty
            
        current_location = self.location_sensor.get_last_gym_observation()
        closest_waypoint_idx = 0
        closest_waypoint_dist = float("inf")
        for i, waypoint in enumerate(self.manuverable_waypoints):
            dist = np.linalg.norm(waypoint.location - current_location)
            if dist < closest_waypoint_dist:
                closest_waypoint_dist = dist
                closest_waypoint_idx = i
        
        # Project vehicle location to waypoints, and look "forward"
        target_waypoint_idx = closest_waypoint_idx
        target_point = self.manuverable_waypoints[target_waypoint_idx].location
        target_waypoint_remaining_dist = self.lookahead_distance - closest_waypoint_dist
        while target_waypoint_remaining_dist > 0:
            next_waypoint_idx = target_waypoint_idx + 1
            next_waypoint_location = self.manuverable_waypoints[next_waypoint_idx % len(self.manuverable_waypoints)].location
            
            vector_to_next_waypoint = next_waypoint_location - target_point
            unit_vector_to_next_waypoint = vector_to_next_waypoint / np.linalg.norm(vector_to_next_waypoint)
            dist_to_next_waypoint = np.linalg.norm(next_waypoint_location - target_point)
            if dist_to_next_waypoint > target_waypoint_remaining_dist:
                target_point = target_point + target_waypoint_remaining_dist * unit_vector_to_next_waypoint
                break
            else:
                target_waypoint_remaining_dist -= dist_to_next_waypoint
                target_point = next_waypoint_location
                target_waypoint_idx = next_waypoint_idx
        
        # Project velocity to the "forward" vector
        velocity_3d = self.velocimeter_sensor.get_last_gym_observation()
        delta_vector = target_point - current_location
        delta_vector_unit = delta_vector / np.linalg.norm(delta_vector)
        projected_velocity_scalar = np.inner(velocity_3d, delta_vector_unit)
        
        normalized_rew = projected_velocity_scalar / 40.0
        if normalized_rew < 0:
            return np.exp(normalized_rew) # Gaussian-like penalty for going backwards
        else:
            return normalized_rew + 1

    def is_terminated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        collision_impulse = self.collision_sensor.get_last_observation().impulse_normal
        collision_impulse_norm = np.linalg.norm(collision_impulse)
        if collision_impulse_norm >= self.collision_threshold:
            return True
        return False
    
    def is_truncated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        return False
