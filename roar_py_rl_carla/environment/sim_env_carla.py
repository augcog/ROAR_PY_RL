from typing import List, Optional
from gymnasium.core import Env
from roar_py_interface import RoarPyActor, RoarPyWaypoint, RoarPyWorld, RoarPyLocationInWorldSensor, RoarPyCollisionSensor, RoarPyVelocimeterSensor
from roar_py_rl import RoarRLEnv, RoarRLSimEnv
from roar_py_carla import RoarPyCarlaVehicle, RoarPyCarlaWorld
from typing import Any, Dict, SupportsFloat, Tuple, Optional
import gymnasium as gym
import numpy as np
import asyncio

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = gym.spaces.flatten_space(self.env.action_space)

    def action(self, action: Any) -> Any:
        return gym.spaces.unflatten(self.env.action_space, action)
    
class RoarRLCarlaSimEnv(RoarRLSimEnv):
    def reset_vehicle(self) -> None:
        # assert isinstance(self.roar_py_actor, RoarPyCarlaVehicle)
        # assert isinstance(self.roar_py_world, RoarPyCarlaWorld)
        spawn_points = self.roar_py_world.spawn_points
        next_spawn_loc, next_spawn_rpy = spawn_points[np.random.randint(len(spawn_points))]
        next_spawn_loc += np.array([0, 0, 1.0])
        
        self.roar_py_actor.set_transform(next_spawn_loc, next_spawn_rpy)
        self.roar_py_actor.set_linear_3d_velocity(np.zeros(3))
        self.roar_py_actor.set_angular_velocity(np.zeros(3))
        async def wait_for_world_ticks(ticks : int) -> None:
            for _ in range(ticks):
                await self.roar_py_world.step()
        asyncio.get_event_loop().run_until_complete(
            wait_for_world_ticks(60)
        )
        
