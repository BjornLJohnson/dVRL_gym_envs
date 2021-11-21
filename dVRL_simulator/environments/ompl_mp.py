import time
import numpy as np
from ompl import base as ob
from ompl import util as ou
from ompl import geometric as og


# target_state: target joint angles
def plan_path(start_state, target_state, planner_type, run_time, isStateValid, output_file=None):
    assert start_state.shape == target_state.shape
    num_states = start_state.shape[0]

    # create the state space
    space = ob.RealVectorStateSpace(num_states)

    # set the bounds of the states
    bounds = ob.RealVectorBounds(num_states)
    low_bdd = np.array([-70., -50., 0., -260., -80., -90.])  # TODO: the bounds of joint 6
    high_bdd = np.array([70., 50., .235, 260., 80., 90.])  # TODO: the bounds of joint 6
    low_bdd[[0, 1, 3, 4, 5]] = low_bdd[[0, 1, 3, 4, 5]] * np.pi / 180.
    high_bdd[[0, 1, 3, 4, 5]] = high_bdd[[0, 1, 3, 4, 5]] * np.pi / 180.
    for i in range(num_states):
        bounds.setLow(i, low_bdd[i])
        bounds.setHigh(i, high_bdd[i])
    space.setBounds(bounds)

    # state information instance for the state space
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    si.setup()

    # start state
    start = ob.State(space)
    print('start')
    for i in range(num_states):
        start()[i] = start_state[i]
        print(start()[i])

    # goal state
    goal = ob.State(space)
    print('goal')
    for i in range(num_states):
        goal()[i] = target_state[i]
        print(goal()[i])

    # problem instance
    pdef = ob.ProblemDefinition(si)

    # set the start and goal states
    pdef.setStartAndGoalStates(start, goal)

    # optimization objective
    objective = ob.PathLengthOptimizationObjective(si)
    pdef.setOptimizationObjective(objective)

    # optimal planner
    optimizing_planner = allocatePlanner(si, planner_type)
    optimizing_planner.setProblemDefinition(pdef)
    optimizing_planner.setup()

    # attempt to solve the planning problem in the given run time
    start_time = time.time()
    solved = optimizing_planner.solve(run_time)
    solved_time = time.time() - start_time

    if solved and pdef.hasExactSolution():
        # Output the length of the path found
        print('{0} found solution of path length {1:.4f} with an optimization objective value of {2:.4f}'.format(
            optimizing_planner.getName(), pdef.getSolutionPath().length(),
            pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

        # output the path as a matrix
        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write(pdef.getSolutionPath().printAsMatrix())

        # get the solution path
        ompl_path = pdef.getSolutionPath()
        path = np.zeros((ompl_path.getStateCount(), 6))
        for i in range(path.shape[0]):
            state = ompl_path.getState(i)
            path[i, :] = np.array([state[j] for j in range(6)])

        return 1., pdef.getSolutionPath().length(), pdef.getSolutionPath().cost(
            pdef.getOptimizationObjective()).value(), solved_time, path
    else:
        print("No solution found.")
        return 0., 0., 0., solved_time, None


def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    elif plannerType.lower() == "rrtconnect":
        return og.RRTConnect(si)
    elif plannerType.lower() == "sbl":
        return og.SBL(si)
    elif plannerType.lower() == "psbl":
        return og.pSBL(si)
    elif plannerType.lower() == "bkpiece":
        return og.BKPIECE1(si)
    elif plannerType.lower() == "lbkpiece":
        return og.LBKPIECE1(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


def pathInterpolation(start, goal, point_num):
    state_dim = len(start)
    path_interpolate = np.zeros((point_num + 2, state_dim))
    diff = goal - start
    path_interpolate[0] = start
    path_interpolate[-1] = goal
    for i in range(1, point_num + 1, 1):
        path_interpolate[i] = path_interpolate[i - 1] + diff / (point_num + 1)

    return path_interpolate
