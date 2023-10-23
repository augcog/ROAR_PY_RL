from typing import Any, SupportsFloat, Tuple, Optional, Set, Dict
from env_util import initialize_roar_env
import gymnasium as gym
import nest_asyncio
import asyncio
import gymnasium.utils.play as gym_play

throttle_speed = 0.8

keys_to_action = {
    "w" : {
        "throttle": throttle_speed,
        "steer": 0.0
    },
    "s" : {
        "throttle": -1.0,
        "steer": 0.0
    },
    "a" : {
        "throttle": 0.0,
        "steer": -1.0
    },
    "d" : {
        "throttle": 0.0,
        "steer": 1.0
    },
    "wa" : {
        "throttle": throttle_speed,
        "steer": -1.0
    },
    "wd" : {
        "throttle": throttle_speed,
        "steer": 1.0
    },
    "sa" : {
        "throttle": -1.0,
        "steer": -1.0
    },
    "sd" : {
        "throttle": -1.0,
        "steer": 1.0
    }
}

class DebugWrapper(gym.Wrapper):
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, rew, terminated, truncated, info = super().step(action)
        #print("rew: ", rew)
        return obs, rew, terminated, truncated, info

def main():
    env = asyncio.run(initialize_roar_env())
    env = DebugWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=400)
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)
    gym_play.play(
        env,
        fps = 20,
        keys_to_action = keys_to_action,
        noop = {
            "throttle": 0.0,
            "steer": 0.0
        }
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    main()
