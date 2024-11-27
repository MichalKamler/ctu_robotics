import cv2 as cv
import numpy as np
import os
from ctu_crs import CRS97
from basler_camera import BaslerCamera

def fitAngleWithinLimits(a1, a2, a_lower_lim, a_upper_lim):
    a = a1 + a2
    
    # Check if already within limits
    if a_lower_lim <= a <= a_upper_lim:
        return a

    # Adjust angle by adding/subtracting 2*pi until it's in the range or not fixable
    if a < a_lower_lim:
        while a < a_lower_lim:
            a += 2 * np.pi
        if a_lower_lim <= a <= a_upper_lim:
            return a
    elif a > a_upper_lim:
        while a > a_upper_lim:
            a -= 2 * np.pi
        if a_lower_lim <= a <= a_upper_lim:
            return a
    
    # Return original angle1 if adjustment fails
    print("Unable to fix angle within limits.")
    return a1

def fitSumOfq(robot, q0 ,q1):
    q_min = robot.q_min
    q_max = robot.q_max
    for i in range(len(q0)):
        q0[i] = fitAngleWithinLimits(q0[i], q1[i], q_min[i], q_max[i])
    return q0
    
def filterSolOutBounds(robot, q_sol):
    valid_q_sol = []
    q_min = robot.q_min
    q_max = robot.q_max
    for i in range(len(q_sol)):
        valid = True
        for j in range(len(q_sol[i])):
            if q_sol[i][j]<q_min[j] or q_sol[i][j]>q_max[j]:
                valid = False
        if valid:
            valid_q_sol.append(q_sol[i])
    return valid_q_sol

def validIk(robot, pose):
    q0 = robot.get_q()
    possible_q_config = robot.ik(pose)
    valid_q_sol = filterSolOutBounds(robot, possible_q_config)
    best_q = None
    if len(valid_q_sol)>0:
        best_q = min(valid_q_sol, key=lambda q: np.linalg.norm(q - q0))
    else:
        print("No solution found for the given x y z")
    return best_q

def end(robot):
    print("end is running!")
    robot.soft_home()
    robot.close()
    monitor_terminal_cmds = False
    print("end completed!")

def start(robot):
    q0 = robot.get_q()
    q = fitSumOfq(robot, q0, [0.0, 0.0, -np.pi/4, 0.0, -np.pi/4, 0.0])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    # moveGripperXYZ(robot, 0, 0, -0.2)
    # setGripperDown(robot,-90)

def moveGripperXYZ(robot, x, y, z):
    print("moveGripperXYZ is running")
    q0 = robot.get_q()
    pose = robot.fk(q0)

    pose[:3, 3] += np.array([x, y, z])

    best_q = validIk(robot, pose)
    if best_q is None:
        return 0
    robot.move_to_q(best_q)
    robot.wait_for_motion_stop()
    print("moveGripperXYZ completed")
    return 1

def reset(robot):
    print("reset is running!")
    robot.reset_motors()
    print("reset completed")

def cage(robot):
    print("cage is running!")
    if robot.in_motion():
        robot.wait_for_motion_stop()
    robot.release()
    print("robot released - OK to go in cage")
    print("cage completed")

def moveBase(robot, deg):
    print("moveBase is running")
    rad = np.deg2rad(deg)
    q0 = robot.get_q()
    q = fitSumOfq(robot, q0, [rad, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    print("moveBase completed")

def rotateGripper(robot, deg):
    print("rotateGripper is running")
    rad = np.deg2rad(deg)
    q0 = robot.get_q()
    q = fitSumOfq(robot, q0, [0.0, 0.0, 0.0, 0.0, 0.0, rad])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    print("rotateGripper completed")

def waitForImg(camera):
    img = camera.grab_image()
    cnt = 0
    while img is None or img.size<=0:
        if cnt >= 20:
            return None
        cnt += 1
        print("Image not captured, staying in while loop")
        img = camera.grab_image()
    return img

def getCamera():
    camera: BaslerCamera = BaslerCamera()
    camera.connect_by_name("camera-crs97")
    camera.open()
    camera.set_parameters()
    camera.start()
    return camera

def endCamera(camera):
    camera.close()

def resetCamera(camera):
    endCamera(camera)
    cam = getCamera()
    return cam