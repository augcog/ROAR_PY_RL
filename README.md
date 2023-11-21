# Reniforcement Learning Platform and Sample Solutions for Robot Open Autonomous Racing(ROAR) Simulation Race

> Contributors: [Yunhao Cao](https://www.quantumcookie.xyz/), [Tianlun Zhang](https://www.linkedin.com/in/tianlun-zhang/), [Franco Huang](https://www.linkedin.com/in/franco-leonardo-huang-246b83221/), [Aman Saraf](https://www.linkedin.com/in/amansrf/), [Grace Zhang](https://www.linkedin.com/in/grace-z-439b1a1a7/).

This repository is the implementation code of RL platform and a RL solution for [ROAR Simulation Race](https://roar.berkeley.edu/simulation-racing/) based on [ROAR_PY](https://github.com/augcog/ROAR_PY) platform. The solution is inherited from [ROAR-RL-Racer](https://github.com/amansrf/ROAR-RL-Racer) which runs on previous ROAR platform and you can find the technical blog [here](https://roar.berkeley.edu/roar-end-to-end-reinforcement-learning/).

The current solution is trained and evaluated on the [Monza Map](https://roar.berkeley.edu/monza-map/).

![straight](./straight.gif)
![turning](./turning.gif)

[Click here for a longer video (5 mins)](https://youtu.be/NHRImZHa2rk?si=6Auj3D62ioQ3612B)

### Setup
Please follow the setup tutorial [here](https://roar.gitbook.io/roar_py_rl-documentation/installation) to install the dependencies.

Please download the latest maps from ROAR [official website](https://roar.berkeley.edu/berkeley-major-map/)

### Observation Space
#### General
The observation space provided to the agent involves:
1. All sensors attached to the vehicle instance:
    - Basically the `RoarRLSimEnv` will take in an instance of `RoarPyActor` and the observation space of the environment will be a superset of that `RoarPyActor`'s `get_gym_observation_spec()`
    - In ROAR's internal RL code, we added the following sensors to the actor:
        - local coordinate velocimeter
        - gyroscope (angular velocity sensor)

2. Waypoint Information Observation

    Instead of inputting an entire occupancy map into the network, we directly feed numerical information about waypoints near the vehicle as the observation provided to the agent. This is how it works:
    1. During initialization of the environment we specify an array of relative distances (that is an array of floating point values) we want to trace for waypoint information observations
    2. Then in each step we perform trace one by one, and storing them inside `waypoints_information` key in the final observation dict.

    A visualization of waypoint information is below, where the arrow represents the position and heading of the vehicle, the bule points represent centered waypoints, and the red points represent boundries. 
    ![visualization of waypoint information](https://github.com/augcog/ROAR_PY_RL/blob/main/visual.png)

#### Our solution
The observation space active in our solution includes:
- Velocity Vector (3x1)
- IMU Vector (3x1)
- Waypoint Information (9x4)

For more details, please refer to [Observation Space documentation](https://roar.gitbook.io/roar_py_rl-documentation/environment-details/sim-environments/observation-space)

### Action Space
The action space of every `RoarRLEnv` would be identical to the return value of `RoarPyActor.get_action_spec()`. 

The action space provided to the agent involves:
- throttle
- steering
- brake
- hand_brake
- reverse

For more details, please refer to [Action Space documentation](https://roar.gitbook.io/roar_py_rl-documentation/environment-details/action-space)

##### Our solution
The action space active in our solution includes:
- Throttle: Box(-1.0, 1.0, (1, ), float32)
- Steering: Box(-1.0, 1.0, (1, ), float32)

### Reward Function

The reward function includes:
1. rewards for traveled distance (to smooth out Q function)
2. penalty for collisions
3. penalty for distance away from the centered waypoints

For detailed rewards caculation, please refer to [Reward Fucntion documentation](https://roar.gitbook.io/roar_py_rl-documentation/environment-details)

### Train
To run the training of our method, you need to:
1. Modify wandb setup and other hyperparameters in `training/train_online.py`.
2. Move into training folder and run the training script.
```bash
cd training
python train_online.py
```

### Eval
The models are stored under `training/models` by default. After you have a trained model, you can go to `training/eval_agent.py` to modify the model path and run this script for evaluation. 
```bash
python training/eval_agent
```

### List of References

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [Soft Actor Critic for Discrete Action Settings](https://arxiv.org/abs/1910.07207)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [The Bellman Error is a Poor Replacement for Value Error](https://arxiv.org/abs/2201.12417)
- [Correcting Robot Plans with Natural Language Feedback](https://arxiv.org/abs/2204.05186)
- [Variable Decision-Frequency Option Critic](https://arxiv.org/abs/2212.04407)
- [Decision Transformer - Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
- [Subwords as Skills Tokenization for Sparse-Reward Reinforcement Learning](https://arxiv.org/abs/2309.04459)
- [Champion Level Drone Racing With DRL](https://www.nature.com/articles/s41586-023-06419-4)
