from pyrep.pyrep import PyRep
from dVRL_simulator.environments.BasePSMEnv import BasePSMEnv, PSMSimObject, PSMSimRefFrame
import numpy as np
import dVRL_simulator.environments.utils as utils
from dVRL_simulator.vrep.ArmPSM_pyrep import ArmPSM
from dVRL_simulator.vrep.simObjects_pyrep import needle, target_pose, collisionCheck, collision4MP


def objectInit(pr):
    armPSM1 = ArmPSM(pr, armNumber=1)
    armPSM2 = ArmPSM(pr, armNumber=2)
    needle_obj = needle(pr)
    target_pose_obj = target_pose(pr)

    armPSM1.getPoseAtEE()
    armPSM1.getJointAngles()
    # initialization of reading proximity sensors
    # armPSM1.readProximitySensors()
    # initialization of getting ee pose in arm1 and arm2 base frame
    armPSM1.EE_virtual_handle.get_pose(relative_to=armPSM1.base_handle)
    armPSM1.EE_virtual_handle.get_pose(relative_to=armPSM2.base_handle)
    # initialization of getting base pose in arm2 base frame
    armPSM1.base_handle.get_pose(relative_to=armPSM2.base_handle)
    # initialization of getting ee pose in arm2 ee frame
    armPSM1.EE_virtual_handle.get_pose(relative_to=armPSM2.EE_virtual_handle)

    armPSM2.getPoseAtEE()
    armPSM2.getJointAngles()
    # initialization of reading proximity sensors
    #     armPSM2.readProximitySensors()
    # initialization of getting ee pose in arm1 and arm2 base frame
    armPSM2.EE_virtual_handle.get_pose(relative_to=armPSM1.base_handle)
    armPSM2.EE_virtual_handle.get_pose(relative_to=armPSM2.base_handle)
    # initialization of getting base pose in arm1 base frame
    armPSM2.base_handle.get_pose(relative_to=armPSM1.base_handle)
    # initialization of getting ee pose in arm1 ee frame
    armPSM2.EE_virtual_handle.get_pose(relative_to=armPSM1.EE_virtual_handle)

    needle_obj.getPose(armPSM1.base_handle)
    needle_obj.getPose(armPSM2.base_handle)
    # initialization of getting pose in arm1 and arm2 ee frame
    needle_obj.getPose(armPSM1.EE_virtual_handle)
    needle_obj.getPose(armPSM2.EE_virtual_handle)

    # collision detection
    collision1 = collisionCheck(pr, 1)
    collision2 = collisionCheck(pr, 2)
    collision_mp1 = collision4MP(pr, 1)
    collision_mp2 = collision4MP(pr, 2)

    pr.step()

    return armPSM1, armPSM2, needle_obj, target_pose_obj, collision1, collision2, collision_mp1, collision_mp2


class PyRepPSMEnv(BasePSMEnv):
    def __init__(self, scene_path, multi_goal=False, headless=True):
        super(PyRepPSMEnv, self).__init__(multi_goal=multi_goal)

        self.pr = PyRep()
        self.pr.launch(scene_path, headless=headless)
        self.pr.start()

        # initializations of the objects and the get functions
        self.psm_move, self.psm_stay, self.needle_obj, self.target_pose_obj, self.collision1, self.collision2, \
            self.collision_mp_move, self.collision_mp_stay = objectInit(self.pr)

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

    def get_handle(self, refFrameEnum):
        """

        Parameters
        ----------
        refFrameEnum: which handle we care about

        Returns
        -------
        The object handle desired
        """
        if refFrameEnum is PSMSimRefFrame.EE_STAY:
            return self.psm_stay.EE_virtual_handle
        elif refFrameEnum is PSMSimRefFrame.EE_MOVE:
            return self.psm_move.EE_virtual_handle
        elif refFrameEnum is PSMSimRefFrame.BASE_STAY:
            return self.psm_stay.base_handle
        elif refFrameEnum is PSMSimRefFrame.BASE_MOVE:
            return self.psm_stay.base_handle
        elif refFrameEnum is PSMSimRefFrame.NEEDLE:
            return self.needle_obj.needle_handle
        elif refFrameEnum is PSMSimRefFrame.WORLD:
            return None
        else:
            raise ValueError("Invalid desired reference frame")

    def needle_is_grasped(self):
        """

        Returns
        -------
        Whether or not needle is successfully grasped by one of the arms
        """
        for _ in range(5):
            self.pr.step()

        return self.needle_obj.isGrasped()

    def get_joint_angles(self, simObjectEnum):
        """

        Parameters
        ----------
        simObjectEnum: which arm we care about

        Returns
        -------
        The joint angles of the arm
        """
        if simObjectEnum is PSMSimObject.PSM_STAY:
            return self.psm_stay.getJointAngles()
        elif simObjectEnum is PSMSimObject.PSM_MOVE:
            return self.psm_move.getJointAngles()
        else:
            raise ValueError("Can only get joint angles for the psm arms")

    def set_joint_angles(self, simObjectEnum, jointAngles, jawAngle=np.pi / 4):
        """

        Parameters
        ----------
        simObjectEnum: which arm we care about
        jointAngles: the joint angles to set the arm to
        jawAngle:  what angle to set the jaw to

        """
        if simObjectEnum is PSMSimObject.PSM_STAY:
            self.psm_stay.setJointAngles(jointAngles, jawAngle)
        elif simObjectEnum is PSMSimObject.PSM_MOVE:
            self.psm_move.setJointAngles(jointAngles, jawAngle)
        else:
            raise ValueError("Can only set joint angles for the psm arms")

        for _ in range(5):
            self.pr.step()

    def get_object_pose(self, simObjectEnum, refFrameEnum):
        """

        Parameters
        ----------
        simObjectEnum: which  object in the scene we care about
        refFrameEnum: which reference frame to get the pose with respect to

        Returns            return pose[:3], pose[3:]
        -------
        The pose of the object in the desired frame
        """

        relative_handle = self.get_handle(refFrameEnum)
        if simObjectEnum is PSMSimObject.PSM_STAY:
            pose = self.psm_stay.EE_virtual_handle.get_pose(relative_handle)
            pos = pose[:3]
            quat = pose[3:]
        elif simObjectEnum is PSMSimObject.PSM_MOVE:
            pose = self.psm_move.EE_virtual_handle.get_pose(relative_handle)
            pos = pose[:3]
            quat = pose[3:]
        elif simObjectEnum is PSMSimObject.NEEDLE_OBJ:
            pos, quat = self.needle_obj.getPose(relative_handle)
        else:
            raise ValueError("Invalid sim object selection")

        return pos, utils.standardize_quat(quat)

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
        relative_handle = self.get_handle(refFrameEnum)
        if simObjectEnum is PSMSimObject.PSM_STAY:
            self.psm_stay.setPoseAtEE(pose[:3], pose[3:], jaw_angle, relative_handle)
        elif simObjectEnum is PSMSimObject.PSM_MOVE:
            self.psm_move.setPoseAtEE(pose[:3], pose[3:], jaw_angle, relative_handle)
        elif simObjectEnum is PSMSimObject.NEEDLE_OBJ:
            self.needle_obj.setPose(pose[:3], pose[3:], relative_handle)
        else:
            raise ValueError("Invalid sim object selection")

        if step_env:
            for _ in range(5):
                self.pr.step()

    def check_collision(self):
        return self.collision_mp_move.checkCollision()
