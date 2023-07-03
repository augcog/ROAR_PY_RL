import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
import roar_py_rl_carla
import roar_py_carla
import roar_py_interface
import carla
import wandb
from wandb.integration.sb3 import WandbCallback
import asyncio
import nest_asyncio

async def initialize_env():
    carla_client = carla.Client('localhost', 3000)
    carla_client.set_timeout(15.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.01)
    world.set_asynchronous(False)
    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        *world.spawn_points[0],
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
        50,
        50,
        5.0,
        5.0,
        name="occupancy_map"
    )
    velocimeter_sensor = vehicle.attach_velocimeter_sensor("velocimeter")
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
    env = roar_py_rl_carla.FlattenActionWrapper(env)
    env = gym.wrappers.FilterObservation(env, ["occupancy_map", "velocimeter"])
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TimeLimit(env, 600)
    env = gym.wrappers.RecordVideo(env, "videos/")
    return env

def main():
    wandb_run = wandb.init(
        project="Major_Map_Debug_ROAR_PY",
        entity="roar",
        name="First Run",
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )
    env = asyncio.run(initialize_env())
    model = SAC(
        "MlpPolicy",
        env,
        optimize_memory_usage=True,
        replay_buffer_kwargs={
            "handle_timeout_termination": False
        }
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