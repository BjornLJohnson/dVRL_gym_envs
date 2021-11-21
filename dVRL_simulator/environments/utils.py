import math
import numpy as np
import transforms3d.quaternions as quaternions
import transforms3d.affines as affines

standard_quat = np.array([-0.626531, -0.619092, -0.33976, -0.329681])


def poses_are_close(p1, p2, max_pos_diff, max_angle_diff):
    assert p1.shape[0] == p2.shape[0]

    pos_diff = np.linalg.norm(p1[:3] - p2[:3])
    angle_diff = quat_angle_diff(p1[3:], p2[3:])
    return pos_diff <= max_pos_diff and angle_diff <= max_angle_diff


def distance_reward(achieved_goal, goal, reward_info):
    spatial_diff = achieved_goal[:3] - goal[:3]
    spatial_dist = np.linalg.norm(spatial_diff)

    angle_dist = np.abs(quat_angle_diff(achieved_goal[3:], goal[3:]))

    return reward_info['step'] + reward_info['pos_diff_weight'] * spatial_dist + \
           reward_info['angle_diff_weight'] * angle_dist


# region Actions and Poses

def get_next_pose_from_action(pos_move_stay, quat_move_stay, action):
    # apply position and quaternion controls
    next_pos_move_stay = pos_move_stay + action[:3]

    next_quat_move_stay = quat_mult(quat_move_stay, action[3:])

    return next_pos_move_stay, next_quat_move_stay


def get_action(prev_pose, current_pose):
    pos_move_stay, quat_move_stay = current_pose[:3], current_pose[3:]
    # calculate delta pos_move_stay
    delta_pos_move_stay = pos_move_stay - prev_pose[:3]

    # calculate delta quat_move_stay
    prev_quat = prev_pose[3:]
    inv_prev_quat_move_stay = quat_inv(unit(prev_quat))
    delta_quat_move_stay = quat_mult(quat_move_stay, inv_prev_quat_move_stay)

    # check the sign of delta_quat_move_stay
    if np.linalg.norm(-delta_quat_move_stay - standard_quat) <= np.linalg.norm(delta_quat_move_stay - standard_quat):
        delta_quat_move_stay = -delta_quat_move_stay

    return np.concatenate((delta_pos_move_stay, delta_quat_move_stay))


def get_action_euler(prev_pose, current_pose):
    # calculate delta pos_move_stay
    delta_pos_move_stay = current_pose[:3] - prev_pose[:3]
    # calculate delta_eul_move_stay
    delta_eul_move_stay = current_pose[3:] - prev_pose[3:]
    # angle wrap around
    more_idx = np.where(delta_eul_move_stay > np.pi)[0]
    delta_eul_move_stay[more_idx] -= 2 * np.pi
    less_idx = np.where(delta_eul_move_stay < -np.pi)[0]
    delta_eul_move_stay[less_idx] += 2 * np.pi

    return np.concatenate((delta_pos_move_stay, delta_eul_move_stay))


# TODO: Make sure this actually works
def get_action_local(prev_pose, curr_pose):
    q1 = np.zeros((4,))
    q1[0] = prev_pose[-1]
    q1[1:] = prev_pose[3:6]
    q2 = np.zeros((4,))
    q2[0] = curr_pose[-1]
    q2[1:] = curr_pose[3:6]

    q1inv = quaternions.qinverse(q1)
    R1inv = quaternions.quat2mat(q1inv)
    v1 = prev_pose[:3]

    R2 = quaternions.quat2mat(q2)
    v2 = curr_pose[:3]

    T1inv = np.concatenate([np.concatenate([R1inv, np.expand_dims(-R1inv @ v1, axis=1)], axis=1),
                            [[0, 0, 0, 1]]], axis=0)

    T2 = np.concatenate([np.concatenate([R2, np.expand_dims(v2, axis=1)], axis=1),
                         [[0, 0, 0, 1]]], axis=0)
    T12 = T1inv @ T2

    v, R, _, _ = affines.decompose44(T12)
    q = quaternions.mat2quat(R)

    action = np.array([v[0], v[1], v[2], q[1], q[2], q[3], q[0]])
    return action


# endregion Actions and Poses

# region Random Sampling


def sample_needle_pt(aon_range, r_range, t_range, p_range, needle_R, aon_avoid_range=None):
    angle_on_needle = np.random.uniform(aon_range[0], aon_range[1])
    if aon_avoid_range is not None:
        while aon_avoid_range[0] <= angle_on_needle <= aon_avoid_range[1]:
            angle_on_needle = np.random.uniform(aon_range[0], aon_range[1])

    # point and tangent direction in the needle frame
    needle_pt = np.array([0, needle_R * np.cos(angle_on_needle) + needle_R / 2, needle_R * np.sin(angle_on_needle)])
    needle_tan = np.cross([needle_pt[0], needle_pt[1] - needle_R / 2, needle_pt[2]], [1, 0, 0])
    needle_tan /= np.linalg.norm(needle_tan)  # normalize the direction vector

    # grasping point, grasping direction, and normal direction in the needle frame
    r, theta, phi = np.random.uniform(r_range[0], r_range[1]), np.random.uniform(t_range[0], t_range[1]), \
                    np.random.uniform(p_range[0], p_range[1])
    grasp_pt = np.array([r * np.cos(phi), r * np.sin(phi) * np.cos(theta) + needle_pt[1],
                         r * np.sin(phi) * np.sin(theta) + needle_pt[2]])
    grasp_dir = needle_pt - grasp_pt  # from the grasping point to the needle point
    grasp_dir /= np.linalg.norm(grasp_dir)  # normalize the direction vector
    needle_nor = np.cross(-grasp_dir, needle_tan)
    needle_nor /= np.linalg.norm(needle_nor)  # normalize the direction vector

    # grasping point frame: 0 at grasp_pt, y axis aligns to grasp_dir, x axis aligns to needle_nor (same as ee frame!)
    g_pt_frame_z = np.cross(needle_nor, grasp_dir)
    g_pt_frame_z /= np.linalg.norm(g_pt_frame_z)  # normalize the direction vector
    H_needle_grasp_pt = np.mat([[needle_nor[0], grasp_dir[0], g_pt_frame_z[0], grasp_pt[0]], \
                                [needle_nor[1], grasp_dir[1], g_pt_frame_z[1], grasp_pt[1]], \
                                [needle_nor[2], grasp_dir[2], g_pt_frame_z[2], grasp_pt[2]], \
                                [0, 0, 0, 1]])

    return H_needle_grasp_pt, angle_on_needle


# init_pose_ee_mean: (7,), 3d pos and 4d quaternion rotation
# pos_range: positive int, m; angle_range: positive int, rad
def sample_ee_init_pose(init_pose_ee_mean, pos_range, angle_range):
    # Add noise to position
    init_pose_ee = np.zeros(7)  # pos and quat
    init_pose_ee[:3] = np.random.uniform(init_pose_ee_mean[:3] - pos_range, init_pose_ee_mean[:3] + pos_range)

    # Change quaternion to axis-angle rotation representation for easy randomization
    init_pose_axis_angle = quat_to_ax_angle(init_pose_ee_mean[3:])

    # Add noise to angle
    init_angle_ee = np.random.uniform(init_pose_axis_angle[-1] - angle_range, init_pose_axis_angle[-1] + angle_range)

    # Change back to quaterion representation for use with simulator/models
    init_quat_ee = quaternions.axangle2quat(init_pose_axis_angle[:-1], init_angle_ee, is_normalized=True)  # wxyz
    init_pose_ee[3:-1] = init_quat_ee[1:]
    init_pose_ee[-1] = init_quat_ee[0]

    return init_pose_ee


def sample_interpolated_init_pose(init_pose_move_needle_mean, goal_pose_move_needle, difficulty):
    # Scale initialization by difficulty
    pos_range = .01 * difficulty ** 2
    angle_range = 30 * np.pi / 180 * difficulty ** 2

    init_pose_move = np.zeros(7)

    # Scale distance from needle differently from other positioning
    init_pose_move[0] = lerp(goal_pose_move_needle[0], init_pose_move_needle_mean[0], difficulty * 0.5 + 0.5)
    init_pose_move[1:3] = lerp(goal_pose_move_needle[1:3], init_pose_move_needle_mean[1:3], difficulty)

    # use spherical interpolation for quaternion orientation
    init_pose_move[3:] = slerp(goal_pose_move_needle[3:], init_pose_move_needle_mean[3:], difficulty)

    # sample an initial pose for psm_move in needle frame
    init_pose_move_needle = sample_ee_init_pose(init_pose_move, pos_range, angle_range)

    return init_pose_move_needle


def sample_random_quaternion():
    u = np.random.uniform(0, 1, 3)
    c1 = np.sqrt(1 - u[0])
    c2 = np.sqrt(u[0])
    t1 = 2 * np.pi * u[1]
    t2 = 2 * np.pi * u[2]
    q = np.array([c1 * np.sin(t1), c1 * np.cos(t1), c2 * np.sin(t2), c2 * np.cos(t2)])
    return standardize_quat(q)


def sample_random_action(pos_action_bound, quat_action_bound):
    theta_des = np.random.uniform(0, quat_action_bound)
    quat_action = clamp_quat_diff(sample_random_quaternion(), theta_max=theta_des)
    pos_action = np.random.uniform(-1., 1., 3) * pos_action_bound
    return np.concatenate([pos_action, quat_action])


def sample_move_pose_needle_frame():
    quat = sample_random_quaternion()
    x = np.random.uniform(-0.02, 0.01)
    y = np.random.uniform(-0.01, 0.01)
    z = np.random.uniform(-0.01, 0.01)
    pos = np.array([x, y, z])
    return np.concatenate([pos, quat])


# endregion Random Sampling

# region Math

def are_close(vec1, vec2, epsilon):
    return np.linalg.norm(vec1-vec2) < epsilon


def unit(vec):
    return vec / max(np.linalg.norm(vec), 1e-6)


def standardize_quat(q):
    q = unit(q)
    if np.linalg.norm(-q - standard_quat) < np.linalg.norm(q - standard_quat):
        q = -q
    return q


def quat_angle_diff(q1, q2):
    assert q1.shape == (4,)
    assert q2.shape == (4,)

    diff = np.dot(q1, q2) ** 2
    theta = np.arccos(2 * min(diff, 1) - 1)
    return theta


def quat_inv(q):
    assert q.shape == (4,)

    qinv = q
    qinv[:3] = -qinv[:3]
    return qinv


def quat_mult(q1, q2):
    assert q1.shape == (4,)
    assert q2.shape == (4,)

    result = np.zeros((4,))
    result[-1] = q1[-1] * q2[-1] - np.dot(q1[:3], q2[:3])
    result[:3] = q1[-1] * q2[:3] + q2[-1] * q1[:3] + np.cross(q1[:3], q2[:3])

    return standardize_quat(result)


def quat_to_ax_angle(q):
    assert q.shape == (4,)

    quat_wxyz = np.zeros(4)
    quat_wxyz[1:] = q[:-1]
    quat_wxyz[0] = q[-1]
    axis, angle = quaternions.quat2axangle(quat_wxyz)

    ax_angle = np.zeros(4)
    ax_angle[:-1] = np.array(axis)
    ax_angle[-1] = angle

    return ax_angle


def posquat_to_matrix(pos, quat):
    assert pos.shape == (3,)
    assert quat.shape == (4,)
    T = np.eye(4)
    T[0:3, 0:3] = quaternions.quat2mat([quat[-1], quat[0], quat[1], quat[2]])
    T[0:3, 3] = pos

    return np.array(T)


def matrix_to_posquat(T):
    pos = T[0:3, 3]
    quat = quaternions.mat2quat(T[0:3, 0:3])
    quat = np.array([quat[1], quat[2], quat[3], quat[0]])

    return np.array(pos), standardize_quat(quat)


def lerp(v1, v2, t):
    return v1 + (v2 - v1) * t


def slerp(q1, q2, t):
    assert q1.shape == (4,)
    assert q2.shape == (4,)

    dot = np.dot(q1, q2)
    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, 0, 1.0)
    omega = math.acos(dot)
    res = math.sin((1 - t) * omega) / math.sin(omega) * q1 + math.sin(t * omega) / math.sin(omega) * q2

    return standardize_quat(res)


def clamp_quat_diff(q, theta_max, q_ref=np.array([0, 0, 0, 1])):
    q = unit(q)
    theta = quat_angle_diff(q, q_ref)
    if theta < theta_max:
        return q

    t = theta_max / theta
    q_new = slerp(q_ref, q, t)
    return standardize_quat(q_new)

# endregion Math
