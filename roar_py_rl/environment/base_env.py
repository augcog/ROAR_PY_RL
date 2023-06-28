from roar_py_interface import RoarPyActor, RoarPyWaypoint, RoarPyWorld, RoarPyVisualizer
import gymnasium as gym
import numpy as np
from typing import Any, List, Optional, SupportsFloat, Tuple, Dict
import asyncio
from .reward_util import near_quadratic_bound

class RoarRLEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"]
    }
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, actor : RoarPyActor, manuverable_waypoints : List[RoarPyWaypoint], world : Optional[RoarPyWorld] = None, render_mode = "rgb_array") -> None:
        super().__init__()
        self.roar_py_actor = actor
        self.roar_py_world = world
        self.manuverable_waypoints = manuverable_waypoints
        self.render_mode = render_mode
        self.visualizer = RoarPyVisualizer(actor)
    
    def get_reward(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> SupportsFloat:
        raise NotImplementedError
    
    def is_terminated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        raise NotImplementedError
    
    def is_truncated(self, observation : Any, action : Any, info_dict : Dict[str, Any]) -> bool:
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.Space:
        self.roar_py_actor.get_gym_observation_spec()
    
    @property
    def action_space(self) -> gym.Space:
        self.roar_py_actor.get_action_spec()
    
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        action_task_async = self.roar_py_actor.apply_action(action)
        asyncio.get_event_loop().run_until_complete(
            action_task_async
        )
        if self.roar_py_world is not None:
            tick_world_task_async = self.roar_py_world.step()
            asyncio.get_event_loop().run_until_complete(
                tick_world_task_async
            )

        observation_task_async = self.roar_py_actor.receive_observation()
        asyncio.get_event_loop().run_until_complete(
            observation_task_async
        )
        
        observation = self.roar_py_actor.get_last_gym_observation()

        info_dict = {}
        reward = self.get_reward(observation, action, info_dict)
        terminated, truncated = self.is_terminated(observation, action, info_dict), self.is_truncated(observation, action, info_dict)
        return observation, reward, terminated, truncated, info_dict

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        if self.roar_py_world is not None:
            tick_world_task_async = self.roar_py_world.step()
            asyncio.get_event_loop().run_until_complete(
                tick_world_task_async
            )
        
        observation_task_async = self.roar_py_actor.receive_observation()
        asyncio.get_event_loop().run_until_complete(
            observation_task_async
        )
        
        observation = self.roar_py_actor.get_last_gym_observation()
        super().reset(seed=seed, options=options)
        return observation, {}
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.visualizer.render()
        else:
            raise NotImplementedError
        
    def close(self) -> None:
        if not self.roar_py_actor.is_closed():
            self.roar_py_actor.close()