import gym
import pickle
import numpy as np
from gym import spaces
from gym.utils import seeding
from enum import Enum, auto
import dVRL_simulator.environments.utils as utils
from dVRL_simulator.environments.ompl_mp import plan_path, pathInterpolation


class PSMSimObject(Enum):
    PSM_MOVE = auto()
    PSM_STAY = auto()
    NEEDLE_OBJ = auto()


class PSMSimRefFrame(Enum):
    BASE_MOVE = auto()
    BASE_STAY = auto()
    EE_MOVE = auto()
    EE_STAY = auto()
    NEEDLE = auto()
    WORLD = auto()


class BasePSMEnv(gym.GoalEnv):
    def __init__(self, multi_goal=False):

        # Environment information
        self.action_dim = 7
        self.state_dim = 21
        self.goal_dim = 7
        self.T = 50

        self.needle_R = 0.0054

        self.action_scaling = np.array([2.2e-3, 2.2e-3, 2.2e-3, 3e-2, 3e-2, 3e-2, 1.1])
        self.quat_action_bound = np.pi / 8

        # Reward samples and success criteria
        self.reward_info = {
            'collide': -1.0,
            'success': 1.0,
            'step': -0.02,
            'pos_diff_weight': -0.5,  # initial distance is up to ~0.03
            'angle_diff_weight': -0.1 / (2 * np.pi),  # max angle diff is 2 pi, reasonable is ~0.5
            'max_pos_diff': 0.002,  # 1e-3 #m
            'max_angle_diff': 0.45  # 0.11 #rad
        }

        # Grasp initialization parameters
        self.multi_goal = multi_goal
        self.discrete_aon_set = [10, 13, 16, 19, 21, 23, 26]
        self.full_aon_range = [10 * np.pi / 18, 26 * np.pi / 18]
        self.move_aon_range = [21 * np.pi / 18, 21 * np.pi / 18]  # Full range is 10-26, ideal is 21
        self.stay_aon_range = [16 * np.pi / 18, 16 * np.pi / 18]  # Full range is 10-26, ideal is 16
        self.t_range_stay = [np.pi, np.pi]  # [0, 2 * np.pi]
        self.p_range_stay = [0, 0]  # [0, .4 * np.pi]
        self.t_range_move = [.5 * np.pi, .5 * np.pi]
        self.p_range_move = [np.pi, np.pi]
        self.r_range = [0.007, 0.007]  # [0.0065, 0.008]

        # Other initialization constants
        self.init_pose_move_needle_mean = np.array([-0.03, 0., 0., -0.64196075, -0.64019813, -0.31923685, 0.27072909])
        self.pose_needle_world = np.array([-1.5388, -6.2888e-02, 7.0149e-01, -0.67231046, -0.67754444, -0.21333313, -0.20837743])
        self.init_pose_stay_base = np.array([-0.11076701, 0.04173517, -0.10018039, 0.26345886, -0.36381617, -0.51663586, -0.72891331])

        # Other sim parameters etc
        self.done = False
        self.current_time_step = 0
        self.init_pos_needle_stay_base = None  # the initial pose of needle in psm_stay base frame
        self.mp_start_state = None  # the initial state (joint angles) for the MP algorithm
        self.mp_target_state = None  # the final state (joint angles) for the MP algorithm
        self.point_num = 21  # number of interpolated configuration
        self.run_mp_prob = 0.0  # the probability of running the MP algorithm after an episode fails
        self.run_mp_prob_diff = 0.00045  # add this to self.run_mp_prob after each episode
        self.mp_algo = 'bitstar'
        self.np_random = None
        self.desired_goal = None
        self.joint_angles_stay = None
        self.jaw_angle_stay = None

        self.seed()
        self._env_setup()

        self.action_space = spaces.Box(-1., 1., shape=(self.action_dim,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype='float32'),
        ))

    def __del__(self):
        self.close()

    # region Env Methods
    # ----------------------------

    # compute reward only based on samples, in batch
    def compute_reward(self, achieved_goal, goal, info):
        assert info['is_success'].shape == info['is_collide'].shape
        rewards = np.zeros(info['is_success'].shape[0]) - .02
        is_success, is_collide = info['is_success'][:, 0].astype(bool), info['is_collide'][:, 0].astype(bool)
        rewards[is_success] = 1.
        rewards[is_collide] = -1.

        return rewards

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, scale_action=True):
        if not self.done:
            if scale_action:
                action = self.scale_action(action)
            self._pre_step_callback(action)
            self._set_action(action)

        obs = self.get_obs()
        if not self.done:
            success = utils.poses_are_close(obs['achieved_goal'], obs['desired_goal'], self.reward_info['max_pos_diff'],
                                            self.reward_info['max_angle_diff'])
            interaction_info = self._interaction_reward(obs['achieved_goal'], obs['desired_goal'])
            reward = interaction_info['reward']
            collide = interaction_info['collide']
        else:
            success = np.array([-1.])
            reward = np.array([-1.])
            collide = np.array([-1.])

        info = {
            'is_success': success,
            'is_collide': collide,
            # 'n_demo_episode': mp_path
        }

        # We return done=False to make sure baselines does not terminate the episode early and break
        done = False
        return obs, reward, done, info

    def reset(self):
        self.done = False
        self.current_time_step = 0
        # reset mp_path
        self.mp_path = None

        init_info = self._full_initialization()

        self.desired_goal = init_info['desired_goal']
        self.mp_target_state = init_info['init_joint_angles_move']
        self.mp_start_state = init_info['goal_joint_angles_move']
        self.init_pos_needle_stay_base = init_info['pose_needle_stay_base'][:3]
        self.joint_angles_stay = init_info['joint_angles_stay']

        # get observation
        return self.get_obs()

    # endregion Env Methods

    # region Utility Methods
    # ----------------------------

    def get_obs(self):
        if self.done:
            return {
                'observation': np.zeros(self.state_dim) - 1.,
                'achieved_goal': np.zeros(int(self.goal_dim)) - 1.,
                'desired_goal': np.zeros(int(self.goal_dim)) - 1.
            }
        # get pose_move_stay, pose_needle_stay, and pose_needle_move
        pos_move_stay, quat_move_stay = self.get_object_pose(PSMSimObject.PSM_MOVE, PSMSimRefFrame.EE_STAY)
        pos_needle_stay, quat_needle_stay = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.EE_STAY)
        pos_needle_move, quat_needle_move = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.EE_MOVE)

        return {
            'observation': np.concatenate(
                (pos_move_stay, quat_move_stay, pos_needle_stay, quat_needle_stay, pos_needle_move, quat_needle_move)),
            'achieved_goal': np.concatenate((pos_needle_move, quat_needle_move)),
            'desired_goal': self.desired_goal.copy()
        }

    def needle_is_grasped(self):
        """

        Returns
        -------
        Whether or not needle is successfully grasped by one of the arms
        """
        raise NotImplementedError()

    def check_collision(self):
        raise NotImplementedError()

    def check_collision_and_needle_pose(self, old_pos_needle):
        # check collision
        collide = self.check_collision()
        # check the pos of the needle
        pos_needle_ref, _ = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.BASE_STAY)
        needle_dis = np.linalg.norm(np.array(pos_needle_ref) - np.array(old_pos_needle))

        return not collide and needle_dis <= 0.001

    def get_joint_angles(self, simObjectEnum):
        """

        Parameters
        ----------
        simObjectEnum: which arm we care about

        Returns
        -------
        The joint angles of the arm
        """
        raise NotImplementedError()

    def set_joint_angles(self, simObjectEnum, jointAngles, jawAngle=np.pi / 4):
        """

        Parameters
        ----------
        simObjectEnum: which arm we care about
        jointAngles: the joint angles to set the arm to
        jawAngle:  what angle to set the jaw to

        """
        raise NotImplementedError()

    def get_object_pose(self, simObjectEnum, refFrameEnum):
        """

        Parameters
        ----------
        simObjectEnum: which  object in the scene we care about
        refFrameEnum: which reference frame to get the pose with respect to

        Returns
        -------
        The pose of the object in the desired frame
        """
        raise NotImplementedError()

    def set_object_pose(self, simObjectEnum, pose, refFrameEnum, jaw_angle=np.pi / 4, step_env=True):
        """

        Parameters
        ----------
        simObjectEnum:  which object in the scene we care about
        pose: the new pose of the object
        refFrameEnum:  the reference  frame in  which the pose is defined
        jaw_angle:  the desired jaw angle of the gripper
        step_env: whether or not to step the simulator after this
        """
        raise NotImplementedError()

    # endregion Utility Methods

    # region Extension Methods
    # ----------------------------

    def unscale_action(self, action):
        action /= self.action_scaling
        return action

    def scale_action(self, action):
        action *= self.action_scaling
        action[3:] = utils.clamp_quat_diff(action[3:], theta_max=self.quat_action_bound)
        return action

    def sample_action(self):
        return utils.sample_random_action(self.action_scaling, self.quat_action_bound)

    def random_step(self):
        return self.step(self.sample_action)

    def pass_needle(self, psm_give=PSMSimObject.PSM_STAY, slow=False):
        psm_take = PSMSimObject.PSM_MOVE if psm_give is PSMSimObject.PSM_STAY else PSMSimObject.PSM_STAY
        base_give_frame = PSMSimRefFrame.BASE_STAY if psm_give is PSMSimObject.PSM_STAY else PSMSimRefFrame.BASE_MOVE
        jaw_angle = np.pi / 18
        joint_angles_give, _ = self.get_joint_angles(psm_give)
        joint_angles_take, _ = self.get_joint_angles(psm_take)
        pos_needle_base_give, quat_needle_base_give = self.get_object_pose(PSMSimObject.NEEDLE_OBJ,
                                                                           base_give_frame)
        pose_needle_base_give = np.concatenate([pos_needle_base_give, quat_needle_base_give])

        # close the gripper of armPSM_take
        if not slow:
            self.set_joint_angles(psm_take, joint_angles_take, jaw_angle)
        else:
            current_jaw_angle = np.pi / 4
            jaw_diff = (current_jaw_angle - jaw_angle) / 10
            for _ in range(10):
                current_jaw_angle -= jaw_diff
                self.set_joint_angles(psm_take, joint_angles_take, current_jaw_angle)

        # set needle to the initial pose
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, pose_needle_base_give, base_give_frame, step_env=False)

        # open the gripper of armPSM_give
        self.set_joint_angles(psm_give, joint_angles_give, np.pi / 4)
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, pose_needle_base_give, base_give_frame)

        return self.needle_is_grasped()

    def sample_goal_collision_config(self):
        goal_init_info = self._init_near_goal()

        pose_needle_stay_base = goal_init_info["pose_needle_stay_base"]
        joint_angles_stay = goal_init_info["joint_angles_stay"]

        # take up to 4 random actions to move away from the actual goal
        for _ in range(np.random.randint(0, 6)):
            action = self.sample_action()
            self._set_action(action)

        return self._apply_random_action_and_check_collision(joint_angles_stay, pose_needle_stay_base)

    def sample_random_collision_config(self):
        self.desired_goal = np.array([0, 0, 0, 0, 0, 0, 1])
        while True:
            self._total_init('demos/total_init')

            # TODO: Use the proper t, p ranges and make sampling more uniform
            success, _ = self._grasp_needle(aon_range=self.stay_aon_range, aon_avoid_range=None,
                                            t_range=self.t_range_stay,
                                            p_range=self.p_range_stay, PSM=PSMSimObject.PSM_STAY)

            if not success:
                continue

            joint_angles_stay, _ = self.get_joint_angles(PSMSimObject.PSM_STAY)

            # sample an initial pose for psm_move in needle frame
            init_pose_move_needle = utils.sample_move_pose_needle_frame()

            # set ee_move to the initial pose
            self.set_object_pose(PSMSimObject.PSM_MOVE, init_pose_move_needle, PSMSimRefFrame.NEEDLE, np.pi / 4)

            pos_needle_stay_base, quat_needle_stay_base = self.get_object_pose(PSMSimObject.NEEDLE_OBJ,
                                                                               PSMSimRefFrame.BASE_STAY)
            pose_needle_stay_base = np.concatenate([pos_needle_stay_base, quat_needle_stay_base])

            return self._apply_random_action_and_check_collision(joint_angles_stay, pose_needle_stay_base)

    def save_init_pose_stay(self, output_file=None):
        # get the joint angles of armPSM_stay
        joint_angles, jaw_angle = self.get_joint_angles(PSMSimObject.PSM_STAY)
        # get the pose of the needle
        pos_needle_base, quat_needle_base = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.BASE_STAY)
        if output_file is not None:
            # write to the file
            pickle.dump([joint_angles, jaw_angle, pos_needle_base, quat_needle_base], open(output_file, 'wb'))
        else:
            return [joint_angles, jaw_angle, pos_needle_base, quat_needle_base]

    def restore_init_pose_stay(self, input_file=None, init_pose=None):
        if input_file is not None:
            # read the file
            [joint_angles, jaw_angle, pos_needle_base, quat_needle_base] = pickle.load(open(input_file, 'rb'))
        else:
            [joint_angles, jaw_angle, pos_needle_base, quat_needle_base] = init_pose.copy()
        # set the joint angles of armPSM_stay
        self.set_joint_angles(PSMSimObject.PSM_STAY, joint_angles, jaw_angle)
        # set the pose of the needle
        pose_needle_stay_base = np.concatenate([pos_needle_base, quat_needle_base])
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, pose_needle_stay_base, PSMSimRefFrame.BASE_STAY)

    def restore_full_init(self, pose_move_stay, pose_needle_stay, initialize_psm_stay=False):
        if initialize_psm_stay:
            # set the pose of psm_stay
            self.set_object_pose(PSMSimObject.PSM_STAY, self.init_pose_stay_base, PSMSimRefFrame.BASE_STAY)

        # set the pose of psm_move
        self.set_object_pose(PSMSimObject.PSM_MOVE, pose_move_stay, PSMSimRefFrame.EE_STAY)

        # set the pose of the needle
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, pose_needle_stay, PSMSimRefFrame.EE_STAY)

    # endregion Extension Methods

    # region Private Methods
    # ----------------------------

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _pre_step_callback(self, action):
        """A custom callback which is executed before the environment steps
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    # region PsmEnv Methods
    # ----------------------------

    def _interaction_reward(self, achieved_goal, goal):
        info = {}
        # check collision and needle pose
        if not self.check_collision_and_needle_pose(self.init_pos_needle_stay_base):
            info["reward"] = self.reward_info['collide']
            info["done"] = True
            info["collide"] = True
            info["success"] = False
            return info

        if utils.poses_are_close(achieved_goal, goal, self.reward_info['max_pos_diff'], self.reward_info['max_angle_diff']):
            info["reward"] = self.reward_info['success']
            info["done"] = True
            info["collide"] = False
            info["success"] = True
            return info
        else:
            # in free space
            reward = utils.distance_reward(achieved_goal, goal, self.reward_info)
            info["reward"] = reward
            info["done"] = False
            info["collide"] = False
            info["success"] = False
            return info

    def _get_state_action(self, prev_pose):
        # get pose_move_stay, pose_needle_stay, and pose_needle_move
        pos_move_stay, quat_move_stay = self.get_object_pose(PSMSimObject.PSM_MOVE, PSMSimRefFrame.EE_STAY)
        pos_needle_stay, quat_needle_stay = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.EE_STAY)
        pos_needle_move, quat_needle_move = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.EE_MOVE)

        state = np.concatenate(
            [pos_move_stay, quat_move_stay, pos_needle_stay, quat_needle_stay, pos_needle_move, quat_needle_move])

        action = utils.get_action(prev_pose, np.concatenate([pos_move_stay, quat_move_stay]))

        return state, action

    def _set_action(self, action):
        pos_move_stay, quat_move_stay = self.get_object_pose(PSMSimObject.PSM_MOVE, PSMSimRefFrame.EE_STAY)
        next_pos_move_stay, next_quat_move_stay = utils.get_next_pose_from_action(pos_move_stay, quat_move_stay, action)
        next_pose_move_stay = np.concatenate([next_pos_move_stay, next_quat_move_stay])
        self.set_object_pose(PSMSimObject.PSM_MOVE, next_pose_move_stay, PSMSimRefFrame.EE_STAY)

    # endregion PsmEnv Methods

    # region Initialization Methods
    # ----------------------------

    def _total_init(self, total_init_file):
        [joint_angles1, jaw_angle1, joint_angles2, jaw_angle2] = pickle.load(open(total_init_file, 'rb'))
        self.set_joint_angles(PSMSimObject.PSM_MOVE, joint_angles1, jaw_angle1)
        self.set_joint_angles(PSMSimObject.PSM_STAY, joint_angles2, jaw_angle2)

    def _arm_joints_init(self):
        joint_angles, jaw_angle = self.get_joint_angles(PSMSimObject.PSM_STAY)
        joint_angles[0] = -joint_angles[0]
        joint_angles[3] = -joint_angles[3]
        joint_angles[5] = -joint_angles[5]
        self.set_joint_angles(PSMSimObject.PSM_STAY, joint_angles, jaw_angle)

        return joint_angles, jaw_angle

    def _grasp_needle(self, aon_range, aon_avoid_range, t_range, p_range, PSM=PSMSimObject.PSM_STAY):
        base_ref_frame = PSMSimRefFrame.BASE_STAY if PSM == PSMSimObject.PSM_STAY else PSMSimRefFrame.BASE_MOVE
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, self.pose_needle_world, PSMSimRefFrame.WORLD, step_env=False)

        pos_ee_base, quat_ee_base, aon = self._sample_grasp_point(aon_range, aon_avoid_range, t_range, p_range,
                                                                  ref_frame=base_ref_frame)

        # set the ee and needle to the target pos and orientation (base frame)
        pose_psm = np.concatenate([pos_ee_base, quat_ee_base])
        self.set_object_pose(PSM, pose_psm, base_ref_frame, jaw_angle=np.pi / 18)

        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, self.pose_needle_world, PSMSimRefFrame.WORLD)

        return self.needle_is_grasped(), aon

    # sample the grasping pos and quat in the arm base frame
    def _sample_grasp_point(self, aon_range, aon_avoid_range, t_range, p_range, ref_frame=PSMSimRefFrame.BASE_STAY):
        # get pos and orn of the needle in the arm base frame
        pos_needle_base, quat_needle_base = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, ref_frame)
        H_arm_base_needle = np.mat(utils.posquat_to_matrix(pos_needle_base, quat_needle_base))

        H_needle_grasp_pt, aon = utils.sample_needle_pt(aon_range, r_range=self.r_range, t_range=t_range,
                                                        p_range=p_range, needle_R=self.needle_R,
                                                        aon_avoid_range=aon_avoid_range)
        H_arm_base_grasp_pt = H_arm_base_needle * H_needle_grasp_pt
        grasp_pos, grasp_quat = utils.matrix_to_posquat(np.squeeze(np.asarray(H_arm_base_grasp_pt)))

        return grasp_pos, grasp_quat, aon

    def _sample_goal_aon_range(self, discrete_goal_set):
        if self.multi_goal and discrete_goal_set:
            goal_choice = np.random.randint(7)
            goal_aon = self.discrete_aon_set[goal_choice]
            goal_aon_range = [goal_aon * np.pi / 18, goal_aon * np.pi / 18]
        elif self.multi_goal:
            goal_aon_range = self.full_aon_range
        else:  # Single fixed goal
            goal_aon_range = self.move_aon_range

        return goal_aon_range

    def _init_near_goal(self, discrete_goal_set=True):
        while True:
            self._total_init('demos/total_init')

            goal_aon_range = self._sample_goal_aon_range(discrete_goal_set)

            # psm_move grasp the needle at the goal pose
            success, aon = self._grasp_needle(aon_range=goal_aon_range, aon_avoid_range=None, t_range=self.t_range_move,
                                              p_range=self.p_range_move, PSM=PSMSimObject.PSM_MOVE)

            if not success:
                continue

            goal_pose_move_stay = self.get_object_pose(PSMSimObject.PSM_MOVE, PSMSimRefFrame.EE_STAY)

            # sample a grasping point for psm_stay
            pos_stay_base, quat_stay_base, _ = self._sample_grasp_point(aon_range=self.stay_aon_range,
                                                                        aon_avoid_range=[aon - 3 * np.pi / 18,
                                                                                         aon + 3 * np.pi / 18],
                                                                        t_range=self.t_range_stay,
                                                                        p_range=self.p_range_stay)

            # set psm_stay to the grasping point
            pose_stay_base = np.concatenate([pos_stay_base, quat_stay_base])
            self.set_object_pose(PSMSimObject.PSM_STAY, pose_stay_base, PSMSimRefFrame.BASE_STAY, np.pi / 4)

            # get the goal pose of psm_move in needle frame
            goal_pos_move_needle, goal_quat_move_needle = self.get_object_pose(PSMSimObject.PSM_MOVE,
                                                                               PSMSimRefFrame.NEEDLE)

            desired_goal_pos, desired_goal_quat = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.EE_MOVE)

            # pass the needle from psm_move to psm_stay
            joint_angles_stay, _ = self.get_joint_angles(PSMSimObject.PSM_STAY)

            # get the goal configuration of psm_move
            goal_joint_angles_move, _ = self.get_joint_angles(PSMSimObject.PSM_MOVE)

            # attempt to pass the needle to ensure this configuration is valid
            success = self.pass_needle(psm_give=PSMSimObject.PSM_MOVE)
            if not success:
                print("Needle not held by stationary arm")
                continue

            pos_needle_stay_base, quat_needle_stay_base = self.get_object_pose(PSMSimObject.NEEDLE_OBJ,
                                                                               PSMSimRefFrame.BASE_STAY)

            return {
                'pose_needle_stay_base': np.concatenate([pos_needle_stay_base, quat_needle_stay_base]),
                'goal_joint_angles_move': goal_joint_angles_move,
                'joint_angles_stay': joint_angles_stay,
                'desired_goal': np.concatenate([desired_goal_pos, desired_goal_quat]),
                'goal_pose_move_needle': np.concatenate([goal_pos_move_needle, goal_quat_move_needle]),
                'goal_pose_move_stay': goal_pose_move_stay
            }

    def _full_initialization(self, discrete_goal_set=True, difficulty=1):
        while True:

            goal_init_info = self._init_near_goal(discrete_goal_set)

            goal_pose_move_needle = goal_init_info["goal_pose_move_needle"]
            goal_pose_move_stay = goal_init_info["goal_pose_move_stay"]
            pose_needle_stay_base = goal_init_info["pose_needle_stay_base"]
            goal_joint_angles_move = goal_init_info["goal_joint_angles_move"]
            joint_angles_stay = goal_init_info["joint_angles_stay"]
            desired_goal = goal_init_info["desired_goal"]

            # sample an initial pose for psm_move in needle frame
            init_pose_move_needle = utils.sample_interpolated_init_pose(self.init_pose_move_needle_mean, goal_pose_move_needle, difficulty)

            # set ee_move to the initial pose
            self.set_object_pose(PSMSimObject.PSM_MOVE, init_pose_move_needle, PSMSimRefFrame.NEEDLE, np.pi / 4)

            # check collision and needle pose
            success = self.check_collision_and_needle_pose(pose_needle_stay_base[:3])

            # get the initial configuration of psm_move
            init_joint_angles_move, _ = self.get_joint_angles(PSMSimObject.PSM_MOVE)

            if success:
                init_info = {'init_joint_angles_move': init_joint_angles_move,
                             'goal_joint_angles_move': goal_joint_angles_move,
                             'joint_angles_stay': joint_angles_stay,
                             'desired_goal': desired_goal,
                             'goal_pose_move_needle': goal_pose_move_needle,
                             'goal_pose_move_stay': goal_pose_move_stay,
                             'pose_needle_stay_base': pose_needle_stay_base,
                             }
                return init_info

    # endregion Initialization Methods

    def _apply_random_action_and_check_collision(self, joint_angles_stay, pose_needle_stay_base):
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, pose_needle_stay_base, PSMSimRefFrame.BASE_STAY)

        pos_move_stay, quat_move_stay = self.get_object_pose(PSMSimObject.PSM_MOVE, PSMSimRefFrame.EE_STAY)
        pose_move_stay = np.concatenate([pos_move_stay, quat_move_stay])

        pos_needle_stay, quat_needle_stay = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.EE_STAY)
        pose_needle_stay = np.concatenate([pos_needle_stay, quat_needle_stay])

        pose_collision_free = self.check_collision_and_needle_pose(pose_needle_stay_base[:3])
        action_collision_free = -1
        action = utils.sample_random_action(self.action_scaling, self.quat_action_bound)
        action = self.scale_action(action)
        if pose_collision_free:
            self._set_action(action)
            action_collision_free = self.check_collision_and_needle_pose(pose_needle_stay_base[:3])

        return joint_angles_stay, pose_move_stay, pose_needle_stay, action, pose_collision_free, action_collision_free

    # endregion Private Methods

    # region Motion Planning
    # ----------------------------

    # state: joint angles
    def _is_state_valid(self, state):
        # get current joint angles of the arm
        joint_angles, jaw_angle = self.get_joint_angles(PSMSimObject.PSM_MOVE)
        # set the joint angles of the arm to state
        self.set_joint_angles(PSMSimObject.PSM_MOVE, state, jaw_angle)

        # check collision
        valid = self.check_collision_and_needle_pose(self.init_pos_needle_stay_base)

        # if collide, set arm_stay and needle to the initial pose
        if not valid:
            print('Not valid!')
            self.set_joint_angles(PSMSimObject.PSM_STAY, self.joint_angles_stay, self.jaw_angle_stay)
            self.set_object_pose(PSMSimObject.NEEDLE_OBJ, self.pose_needle_world, PSMSimRefFrame.WORLD)

        # reset arm_move to current joint angles
        self.set_joint_angles(PSMSimObject.PSM_MOVE, joint_angles, jaw_angle)

        return valid

    def generate_mp_path(self, init_joint_angles_move=None, goal_joint_angles_move=None, angles_stay=None):
        self.jaw_angle_stay = np.pi / 18
        if init_joint_angles_move is None:
            self.reset()
            mp_start_state = self.mp_start_state
            mp_target_state = self.mp_target_state
        # TODO: DELETE THE BELOW LOGIC AND MAKE ONLY ONE INTERFACE?
        else:
            assert goal_joint_angles_move is not None
            assert angles_stay is not None
            mp_start_state = goal_joint_angles_move
            mp_target_state = init_joint_angles_move
            self.joint_angles_stay = angles_stay

        self.set_joint_angles(PSMSimObject.PSM_MOVE, mp_start_state, np.pi / 4)
        self.set_joint_angles(PSMSimObject.PSM_STAY, self.joint_angles_stay, self.jaw_angle_stay)
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, self.pose_needle_world, PSMSimRefFrame.WORLD)

        solved, _, _, _, path = plan_path(mp_start_state, mp_target_state, self.mp_algo, 2., self._is_state_valid)

        if not solved:
            print("OMPL Solution Not Found")
            return False, -1, -1, -1

        # interpolate between start and goal in the path
        total_point_num = 1 + (len(path) - 1) * (self.point_num + 1)
        path_interpolate = np.zeros((total_point_num, 6))
        for i in range(len(path) - 1):
            path_interpolate[(self.point_num + 1) * i:(self.point_num + 1) * (i + 1) + 1] = pathInterpolation(path[i], path[i + 1], self.point_num)

        # reverse the path
        path_interpolate = np.flip(path_interpolate, axis=0)

        # reset to initial configuration
        self.set_joint_angles(PSMSimObject.PSM_STAY, self.joint_angles_stay, jawAngle=self.jaw_angle_stay)
        self.set_object_pose(PSMSimObject.NEEDLE_OBJ, self.pose_needle_world, PSMSimRefFrame.WORLD)
        self.set_joint_angles(PSMSimObject.PSM_MOVE, path_interpolate[0], np.pi / 4)

        prev_pos_move_stay, prev_quat_move_stay = self.get_object_pose(PSMSimObject.PSM_MOVE, PSMSimRefFrame.EE_STAY)
        prev_pose_move_stay = np.concatenate([prev_pos_move_stay, prev_quat_move_stay])
        state, _ = self._get_state_action(prev_pose_move_stay)

        # 1~7: pose_ee1_ee2, 8~14: pose_needle_ee2, 15~21: pose_needle_ee1, 22~28: action_ee1_ee2
        states = np.zeros((len(path_interpolate), 21))
        actions = np.zeros((len(path_interpolate), 7))
        states[0] = state

        init_pose = self.save_init_pose_stay()
        pos_needle_stay_base, _ = self.get_object_pose(PSMSimObject.NEEDLE_OBJ, PSMSimRefFrame.BASE_STAY)

        # move psm_move along the path to record states and actions
        success = True
        for i in range(1, len(path_interpolate)):
            self.set_joint_angles(PSMSimObject.PSM_MOVE, path_interpolate[i], np.pi / 4)

            # check collision of arm1 with others and needle pose
            if not self.check_collision_and_needle_pose(pos_needle_stay_base):
                success = False
                break
            else:
                # record states and actions
                state, action = self._get_state_action(prev_pose_move_stay)
                states[i] = state
                actions[i - 1] = action
                prev_pose_move_stay = state[:7]

        # try to pass the needle
        if success and self.pass_needle():
            print("Success!")
            return True, states, actions, init_pose
        print("OMPL Solution Fails")
        return False, -1, -1, -1

    #     assert states.shape == (self.point_num, self.state_dim)
    #     assert actions.shape == (self.point_num, self.action_dim)
    #
    #     actions /= self.action_bound  # scale the actions
    #
    #     obs = np.reshape(states, [self.point_num, 1, self.state_dim])
    #     acts = np.reshape(actions[:-1], [self.point_num - 1, 1, self.action_dim])
    #     goals = self.desired_goal.repeat(self.point_num - 1, axis=0)
    #     goals = np.reshape(goals, [self.point_num - 1, 1, self.goal_dim])
    #     achieved_goals = np.reshape(states[1:, -self.goal_dim:], [self.T - 1, 1, self.goal_dim])
    #
    #     info_values = {'is_success': np.zeros([self.point_num - 1, 1, 1]),
    #                    'is_collide': np.zeros([self.point_num - 1, 1, 1])}
    #     info_values['is_success'][-1] = 1
    #
    #     return {'obs': obs, 'acts': acts, 'goals': goals, 'achieved_goals': achieved_goals,
    #             'info_values': info_values}

# endregion Motion Planning
