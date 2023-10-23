import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import asyncio
import nest_asyncio
import os
from pathlib import Path
from typing import Optional, Dict
import torch as th
from typing import Dict, SupportsFloat, Union
from env_util import initialize_roar_env
from roar_py_rl_carla import FlattenActionWrapper

RUN_FPS= 20
SUBSTEPS_PER_STEP = 2
training_params = dict(
    learning_rate = 1e-5,  # be smaller 2.5e-4
    #n_steps = 256 * RUN_FPS, #1024
    batch_size=256,  # mini_batch_size = 256?
    # n_epochs=10,
    gamma=0.97,  # rec range .9 - .99 0.999997
    ent_coef="auto",
    target_entropy=-10.0,
    # gae_lambda=0.95,
    # clip_range_vf=None,
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    use_sde=True,
    sde_sample_freq=RUN_FPS * 2,
    # target_kl=None,
    # tensorboard_log=(Path(misc_params["model_directory"]) / "tensorboard").as_posix(),
    # create_eval_env=False,
    # policy_kwargs=None,
    verbose=1,
    seed=1,
    device=th.device('cuda' if th.cuda.is_available() else 'cpu'),
    # _init_setup_model=True,
)

def find_latest_model(root_path: Path) -> Optional[Path]:
    """
        Find the path of latest model if exists.
    """
    logs_path = (os.path.join(root_path, "logs"))
    if os.path.exists(logs_path) is False:
        print(f"No previous record found in {logs_path}")
        return None
    paths = sorted(os.listdir(logs_path), key=os.path.getmtime)
    paths_dict: Dict[int, Path] = {
        int(path.name.split("_")[2]): path for path in paths
    }
    if len(paths_dict) == 0:
        return None
    latest_model_file_path: Optional[Path] = paths_dict[max(paths_dict.keys())]
    return latest_model_file_path

def get_env(wandb_run) -> gym.Env:
    env = asyncio.run(initialize_roar_env(control_timestep=1.0/RUN_FPS, physics_timestep=1.0/(RUN_FPS*SUBSTEPS_PER_STEP)))
    env = gym.wrappers.FlattenObservation(env)
    env = FlattenActionWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=RUN_FPS*30)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, f"videos/{wandb_run.name}_{wandb_run.id}", step_trigger=lambda x: x % 5000 == 0)
    env = Monitor(env, f"logs/{wandb_run.name}_{wandb_run.id}", allow_early_resets=True)
    return env

def main():
    wandb_run = wandb.init(
        project="Major_Map_Debug_ROAR_PY",
        entity="roar",
        name="Testing",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    ) 
    
    env = get_env(wandb_run)

    models_path = f"models/{wandb_run.name}"
    latest_model_path = find_latest_model(models_path)
    
    if latest_model_path is None:
        # create new models
        model = SAC(
            "MlpPolicy",
            env,
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
            optimize_memory_usage=True,
            replay_buffer_kwargs={"handle_timeout_termination": False}
            **training_params
        )

    model.learn(
        total_timesteps=500_000,
        callback=WandbCallback(
            gradient_save_freq=2000,
            model_save_path=f"models/{wandb_run.name}",
            verbose=2,
        ),
        progress_bar=True,
    )

if __name__ == "__main__":
    nest_asyncio.apply()
    main()
