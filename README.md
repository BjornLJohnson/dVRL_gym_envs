# Gym Environments for Reinforcement Learning with the da Vinci Surgical Robotics Platform

### Official Implementation of the environment described in https://ieeexplore.ieee.org/abstract/document/9561673

### Built off environments from https://github.com/ucsdarclab/dVRL

### Requires PyRep https://github.com/stepjam/PyRep

### Usage

```
import numpy as np
import dVLR_simulator

env = gym.make("dVRLHoldNeedle-v1", headless=False)

env.reset()

action = np.random.random(7)
obs, reward, done, info = env.step(action)
```
