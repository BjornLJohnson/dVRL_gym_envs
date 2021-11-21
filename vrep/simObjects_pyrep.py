#!/usr/bin/env python
# coding: utf-8

import numpy as np
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
from dVRL_simulator.vrep.collisions import Collision
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.backend import sim
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy

# Will look to remove this; for now, it is a placeholder class to control a VisionSensor (not the Camera)
class camera(VisionSensor):
    def __init__(self, pr, camera_name, rgb = True):
        self.camera_handle = VisionSensor(camera_name)
        self.rgb = rgb
        
        self.camera_handle.capture_rgb()
        
    def getImage(self):
        data = self.camera_handle.capture_rgb()
        resolution = self.camera_handle.get_resolution()

        if not self.rgb:
            return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0]])
        return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0], 3])

class collisionCheck(Collision):
    def __init__(self, pr,psm_number):
        super(collisionCheck, self).__init__()

        #self.collision_TTs_TableTop = self.getCollisionHandle('PSM{}_TTs_Table'.format(psm_number))
        #self.collision_TTd_TableTop = self.getCollisionHandle('PSM{}_TTd_Table'.format(psm_number))
        self.collision_Needle_TableTop = self.getCollisionHandle('Needle_Table')
        self.collision_Needle_L3_dx_TOOL = self.getCollisionHandle('Needle_L3_dx_TOOL{}'.format(psm_number))
        self.collision_Needle_L3_sx_TOOL = self.getCollisionHandle('Needle_L3_sx_TOOL{}'.format(psm_number))
        self.collision_Needle_L2_TOOL = self.getCollisionHandle('Needle_L2_TOOL{}'.format(psm_number))

        #super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, ignoreError = True, initialize = True)
        #super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, ignoreError = True, initialize = True)
        super(collisionCheck, self).checkCollision(self.collision_Needle_TableTop)
        super(collisionCheck, self).checkCollision(self.collision_Needle_L3_dx_TOOL)
        super(collisionCheck, self).checkCollision(self.collision_Needle_L3_sx_TOOL)
        super(collisionCheck, self).checkCollision(self.collision_Needle_L2_TOOL)
        
    #Returns True if in collision and False if not in collision
    def checkCollision(self):
        #c1 = super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, ignoreError)
        #c2 = super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, ignoreError)
        c3 = super(collisionCheck, self).checkCollision(self.collision_Needle_TableTop)
        c4 = super(collisionCheck, self).checkCollision(self.collision_Needle_L2_TOOL)

        #return c1 or c2 or c3 or c4
        return c3 or c4

    def checkCollisionGripper(self): 
        c1 = super(collisionCheck, self).checkCollision(self.collision_Needle_L3_dx_TOOL)
        c2 = super(collisionCheck, self).checkCollision(self.collision_Needle_L3_sx_TOOL)

        return c1 or c2


class collision4MP(Collision): 
    def __init__(self, pr, psm_number): 
        #super(collision4MP, self).__init__()

        self.collision_arm_others = self.getCollisionHandle('PSM{}_others'.format(psm_number))
        super(collision4MP, self).checkCollision(self.collision_arm_others)

    #Returns True if in collision and False if not in collision
    def checkCollision(self):
        c = super(collision4MP, self).checkCollision(self.collision_arm_others)

        return c

class table(Shape):
    def __init__(self):
        self.table_top_handle = Shape('customizableTable_tableTop')

    def getPose(self, relative_handle):
        pose = self.table_top_handle.get_pose(relative_to = relative_handle)
        pos, quat = pose[0:3], pose[3:]
        return np.array(pos), np.array(quat)

class obj(Object):
    def __init__(self):
        super(obj, self).__init__()

        self.obj_handle = Shape('Object')
        self.dummy_handle = Shape('Object_Dummy')
        
    def posquat2Matrix(self, pos, quat):
        T = np.eye(4)
        T[0:3, 0:3] = quaternions.quat2mat([quat[-1], quat[0], quat[1], quat[2]])
        T[0:3, 3] = pos

        return np.array(T)

    def matrix2posquat(self,T):
        pos = T[0:3, 3]
        quat = quaternions.mat2quat(T[0:3, 0:3])
        quat = [quat[1], quat[2], quat[3], quat[0]]

        return np.array(pos), np.array(quat)

    def setPose(self, pos, quat, relative_handle):
        b_T_d = self.posquat2Matrix(pos, quat)
        d_T_o = np.array([[1, 0, 0, 0], [0,1,0,0], [0,0,1,0.001], [0,0,0,1]])
        pos, quat = self.matrix2posquat(np.dot(b_T_d,d_T_o))

        self.obj_handle.set_pose(list(np.r_[pos, quat]), relative_to = relative_handle)

    def getPose(self, relative_handle):
        pose = self.dummy_handle.get_pose(relative_to = relative_handle)
        pos, quat = pose[0:3], pose[3:]
        return np.array(pos), np.array(quat)

    def getVel(self):
        return self.dummy_handle.get_velocity()

    def removeGrasped(self):
        self.obj_handle.set_parent()

    def isGrasped(self):
        return not (-1 == self.obj_handle.get_parent())

class needle(Shape):
    def __init__(self, pr):
        #super(needle, self).__init__(pr)

        self.needle_handle = Shape('suture_needle_link_respondable')
        self.needle_handle.get_parent()
    
    def posquat2Matrix(self, pos, quat):
        T = np.eye(4)
        T[0:3, 0:3] = quaternions.quat2mat([quat[-1], quat[0], quat[1], quat[2]])
        T[0:3, 3] = pos

        return np.array(T)

    def matrix2posquat(self,T):
        pos = T[0:3, 3]
        quat = quaternions.mat2quat(T[0:3, 0:3])
        quat = [quat[1], quat[2], quat[3], quat[0]]

        return np.array(pos), np.array(quat)
    
    def setPose(self, pos, quat, relative_handle = None):
        b_T_d = self.posquat2Matrix(pos, quat)
        d_T_o = np.array([[1, 0, 0, 0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        pos, quat = self.matrix2posquat(np.dot(b_T_d,d_T_o))

        self.needle_handle.set_pose(list(np.r_[pos, quat]), relative_handle)

    def getPose(self, relative_handle = None):
        pose = self.needle_handle.get_pose(relative_handle)
        pos, quat = pose[0:3], pose[3:]
        return np.array(pos), np.array(quat)

    def getVel(self):
        return sim.simGetObjectVelocity(sim.simGetObjectHandle('suture_needle_link_respondable'))

    def removeGrasped(self):
        self.needle_handle.set_parent()

    def isGrasped(self):
        # return not (-1 == self.needle_handle.get_parent())
        return not (self.needle_handle.get_parent() is None)
    
    def euler2Quat(self, rpy):
        q = euler.euler2quat(rpy[0], rpy[1], rpy[2],  axes = 'rxyz')
        return np.array([q[1], q[2], q[3], q[0]])

    def quat2Euler(self, quat):
        return np.array(euler.quat2euler([quat[3], quat[0], quat[1], quat[2]], axes = 'rxyz'))


#TODO: target pose for the gripper to grasp the needle
class target_pose(Dummy):
    def __init__(self,pr):
        #super(target_pose, self).__init__()

        self.target_pose_handle = Dummy('target_pose')

        self.getPose(None)

    def setPose(self, pos, quat, relative_handle):
        self.target_pose_handle.set_pose(list(np.r_[pos, quat]), relative_to = relative_handle)

    def getPose(self, relative_handle):
        pose = self.target_pose_handle.get_pose(relative_to = relative_handle)
        pos, quat = pose[0:3], pose[3:]
        return np.array(pos), np.array(quat)
    
class target(Shape):
    def __init__(self,psm_number):
        super(target, self).__init__()

        self.target_handle = Shape('Target_PSM{}'.format(psm_number))

        self.getPosition(-1)

    def setPosition(self, pos, relative_handle):
        self.target_handle.set_pose(list(np.r_[pos, [1,0,0,1]]), relative_to = relative_handle)

    def getPosition(self, relative_handle):
        pose = self.target_handle.get_pose(relative_to = relative_handle)
        return np.array(pose[0:3])
    