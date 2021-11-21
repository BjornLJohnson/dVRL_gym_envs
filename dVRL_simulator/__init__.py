from os.path import dirname, join, abspath
import numpy as np
from gym import utils
from gym.envs.registration import register
from dVRL_simulator.environments.PyRepPSMEnv import PyRepPSMEnv
from dVRL_simulator.environments.VRepPSMEnv import VRepPSMEnv
from dVRL_simulator.environments.MLPPSMEnv import MLPPSMEnv
from dVRL_simulator.environments.PsmEnv_Position import PSMEnv_Position


class PSMHoldNeedleEnv(VRepPSMEnv):
    def __init__(self, multi_goal=False):
        super(PSMHoldNeedleEnv, self).__init__(multi_goal=multi_goal, docker_container="needle_pass_vrep_env")
        utils.EzPickle.__init__(self)


class PSMHoldNeedleEnvPyRep(PyRepPSMEnv):
    def __init__(self, multi_goal=False, headless=True):
        SCENE_FILE = join(dirname(abspath(__file__)), 'environments/VRep_scenes/dVRK-oneArm-hold-needle.ttt')
        super(PSMHoldNeedleEnvPyRep, self).__init__(scene_path=SCENE_FILE, multi_goal=multi_goal, headless=headless)
        utils.EzPickle.__init__(self)


class PSMHoldNeedleEnvMLP(MLPPSMEnv):
    def __init__(self, multi_goal=False):
        MODEL_PATH = "training/collision/model"
        super(PSMHoldNeedleEnvMLP, self).__init__(collision_model_path=MODEL_PATH, multi_goal=multi_goal)
        utils.EzPickle.__init__(self)


class PSMPickEnv(PSMEnv_Position):
    def __init__(self, psm_num=1, reward_type='sparse', randomize_obj=False, randomize_ee=False):
        initial_pos = np.array([0, 0, -0.10])

        super(PSMPickEnv, self).__init__(psm_num=psm_num, n_substeps=1, block_gripper=False,
                                         has_object=True, target_in_the_air=True, height_offset=0.0001,
                                         target_offset=[0, 0, 0.005], obj_range=0.025, target_range=0.025,
                                         distance_threshold=0.003, initial_pos=initial_pos, reward_type=reward_type,
                                         dynamics_enabled=False, two_dimension_only=False,
                                         randomize_initial_pos_obj=randomize_obj, randomize_initial_pos_ee=randomize_ee,
                                         docker_container="vrep_ee_pick")

        utils.EzPickle.__init__(self)


class PSMReachEnv(PSMEnv_Position):
    def __init__(self, psm_num=1, reward_type='sparse'):
        initial_pos = np.array([0, 0, -0.11])

        super(PSMReachEnv, self).__init__(psm_num=psm_num, n_substeps=1, block_gripper=True,
                                          has_object=False, target_in_the_air=True, height_offset=0.01,
                                          target_offset=[0, 0, 0], obj_range=0.05, target_range=0.05,
                                          distance_threshold=0.003, initial_pos=initial_pos, reward_type=reward_type,
                                          dynamics_enabled=False, two_dimension_only=False,
                                          randomize_initial_pos_obj=False, randomize_initial_pos_ee=False,
                                          docker_container="vrep_ee_reach")

        utils.EzPickle.__init__(self)


register(
    id='dVRLReach-v0',
    entry_point='dVRL_simulator:PSMReachEnv',
    max_episode_steps=100,
)

register(
    id='dVRLPick-v0',
    entry_point='dVRL_simulator:PSMPickEnv',
    max_episode_steps=100,
)

register(
    id='dVRLHoldNeedle-v0',
    entry_point='dVRL_simulator:PSMHoldNeedleEnv',
    max_episode_steps=50,
)

register(
    id='dVRLHoldNeedle-v1',
    entry_point='dVRL_simulator:PSMHoldNeedleEnvPyRep',
    max_episode_steps=100,
)

register(
    id='dVRLHoldNeedle-v2',
    entry_point='dVRL_simulator:PSMHoldNeedleEnvMLP',
    max_episode_steps=100,
)
