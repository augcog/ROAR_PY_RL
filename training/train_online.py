import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.ppo.ppo import PPO
import roar_py_rl_carla
import roar_py_carla
import roar_py_interface
import carla
import wandb
from wandb.integration.sb3 import WandbCallback
import asyncio
import nest_asyncio
import os
from pathlib import Path
from typing import Optional, Dict
import torch as th
from networks import Atari_PPO_Adapted_CNN
from typing import Dict, SupportsFloat, Union

run_fps= 32
training_params = dict(
    learning_rate = 0.00003,  # be smaller 2.5e-4
    n_steps = 256 * run_fps, #1024
    batch_size=256,  # mini_batch_size = 256?
    # n_epochs=10,
    gamma=0.97,  # rec range .9 - .99 0.999997
    ent_coef=.00,  # rec range .0 - .01
    # gae_lambda=0.95,
    # clip_range_vf=None,
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    # use_sde=True,
    # sde_sample_freq=misc_params["run_fps"]/2,
    # target_kl=None,
    # tensorboard_log=(Path(misc_params["model_directory"]) / "tensorboard").as_posix(),
    # create_eval_env=False,
    # policy_kwargs=None,
    verbose=1,
    seed=1,
    device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
    # _init_setup_model=True,
)

class ROARActionFilter(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._action_space = gym.spaces.Dict({
            "throttle": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
            "steer": gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        })
    def action(self, action: Dict[str, Union[SupportsFloat, float]]) -> Dict[str, Union[SupportsFloat, float]]:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        # action = {
        #     "throttle": [-1.0, 1.0],
        #     "steer": [-1.0, 1.0]
        # }
        real_action = {
            "throttle": np.clip(action["throttle"], 0.0, 1.0),
            "brake": np.clip(-action["throttle"], 0.0, 1.0),
            "steer": action["steer"],
            "hand_brake": 0.0,
            "reverse": 0.0
        }
        return real_action

async def initialize_env():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.01)
    world.set_asynchronous(False)
    vehicle = world.spawn_vehicle(
        "vehicle.dallara.dallara",
        *world.spawn_points[1],
        True,
        "vehicle"
    )
    assert vehicle is not None, "Failed to spawn vehicle"
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3),
        name="collision_sensor"
    )
    assert collision_sensor is not None, "Failed to attach collision sensor"
    occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
        84,
        84,
        5.0,
        5.0,
        name="occupancy_map"
    )
    
    #TODO: Attach next waypoint to observation
    velocimeter_sensor = vehicle.attach_velocimeter_sensor("velocimeter")
    local_velocimeter_sensor = vehicle.attach_local_velocimeter_sensor("local_velocimeter")

    location_sensor = vehicle.attach_location_in_world_sensor("location")
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB, # Specify what kind of data you want to receive
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
        image_width=400,
        image_height=200
    )
    await world.step()
    await vehicle.receive_observation()
    env = roar_py_rl_carla.RoarRLCarlaSimEnv(
        vehicle,
        world.maneuverable_waypoints,
        location_sensor,
        velocimeter_sensor,
        collision_sensor,
        world = world
    )
    env = ROARActionFilter(env)
    env = roar_py_rl_carla.FlattenActionWrapper(env) # TODO: Filter out some actions
    env = gym.wrappers.FilterObservation(env, ["occupancy_map", "local_velocimeter"])
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeLimit(env, 600)
    env = gym.wrappers.RecordVideo(env, "videos/")
    return env

def find_latest_model(root_path: Path) -> Optional[Path]:
    """
        Find the path of latest model if exists.
    """
    logs_path = (root_path / "logs")
    if logs_path.exists() is False:
        print(f"No previous record found in {logs_path}")
        return None
    paths = sorted(logs_path.iterdir(), key=os.path.getmtime)
    paths_dict: Dict[int, Path] = {
        int(path.name.split("_")[2]): path for path in paths
    }
    if len(paths_dict) == 0:
        return None
    latest_model_file_path: Optional[Path] = paths_dict[max(paths_dict.keys())]
    return latest_model_file_path

def main():
    wandb_run = wandb.init(
        project="Major_Map_Debug_ROAR_PY",
        entity="roar",
        name="Testing",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )
    env = asyncio.run(initialize_env())
    policy_kwargs = dict(
        # features_extractor_class = AutoRacingNet,
        features_extractor_class = Atari_PPO_Adapted_CNN,
        features_extractor_kwargs=dict(features_dim=256),
        use_sde=True
    )
    models_path = f"models/{wandb_run.name}"
    latest_model_path = find_latest_model(models_path)
    
    if latest_model_path is None:
        # create new models
        model = SAC(
                    "MlpPolicy",
                    env,
                    policy_kwargs = policy_kwargs,
                    optimize_memory_usage=True,
                    replay_buffer_kwargs={"handle_timeout_termination": False},
                    **training_params
                )
    else:
        # Load the model
        print(latest_model_path)
        model = SAC.load(
                            latest_model_path,
                            env=env,
                            policy_kwargs = policy_kwargs,
                            optimize_memory_usage=True,
                            replay_buffer_kwargs={"handle_timeout_termination": False}
                            **training_params
                        )

    model.learn(
        total_timesteps=10_000,
        callback=WandbCallback(
            gradient_save_freq=5000,
            model_save_path=f"models/{wandb_run.name}",
            verbose=2,
        ),
        progress_bar=True,
    )



if __name__ == "__main__":
    nest_asyncio.apply()
    main()
