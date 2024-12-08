import threading
import numpy as np
from enum import Enum
import time
import cv2 as cv
import os
from ctu_crs import CRS97
from basler_camera import BaslerCamera
from utils import loadCamDist, arucoMarkersFinder, pairUpAruco, locateCenterOfCubes, drawFoundCubes, loadRT


camMatrix, distCoeff = loadCamDist('npz/calibration_ciirc.npz')
R_base2cam, t_base2cam = loadRT('npz/R_t_base2cam.npz')

T_base2cam = np.eye(4)
T_base2cam[:3, :3] = R_base2cam
T_base2cam[:3, 3] = t_base2cam.flatten()
monitor_terminal_cmds = True

def rotXYZ(rx, ry, rz):
    Rx = np.array([[1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R

R_gripper2cube = rotXYZ(np.pi/2, -np.pi/2, -np.pi/2)

def avgAllMeasurements(multiple_measurements):
    num_measurements = len(multiple_measurements)
    num_poses = len(multiple_measurements[0])

    # Initialize a list to store averaged SE(3) matrices
    average_poses = []

    for pose_idx in range(num_poses):
        pose_measurements = [multiple_measurements[measurement][pose_idx] for measurement in range(num_measurements)]
        
        avg_rotation = np.mean([pose[:3, :3] for pose in pose_measurements], axis=0)
        
        U, _, Vt = np.linalg.svd(avg_rotation)
        avg_rotation = U @ Vt
        
        avg_translation = np.mean([pose[:3, 3] for pose in pose_measurements], axis=0)
        
        avg_pose = np.eye(4)
        avg_pose[:3, :3] = avg_rotation
        avg_pose[:3, 3] = avg_translation
        # if avg_pose[1,3]<=0:
        avg_pose[1,3] = avg_pose[1,3] -(0.03)*avg_pose[1,3]
        # else: 
            # avg_pose[1,3] = avg_pose[1,3] + max(-(0.04)*avg_pose[1,3], 0)

        print("adjust is: ", -(0.04)*avg_pose[1,3])

        average_poses.append(avg_pose)

    return average_poses


def locateAllCubes(camera):
    multiple_measurements = []
    for i in range(10):
        img = waitForImg(camera)
        img, allT_base2marker, ids = arucoMarkersFinder(img, camMatrix, distCoeff, 0.036)
        multiple_measurements.append(allT_base2marker)
    allT_base2marker_avg = avgAllMeasurements(multiple_measurements)

    cubesList = []
    if len(ids)>0:
        pairs = pairUpAruco(allT_base2marker_avg, ids)
        for pair in pairs:
            cubes = locateCenterOfCubes(pair)
            img = drawFoundCubes(img, camMatrix, distCoeff, cubes, T_base2cam)
            cubesList.append(cubes)


    if img is None:
        print("Error: Could not load image.")
    else:
        cv.namedWindow("Image Window", cv.WINDOW_NORMAL)
        cv.resizeWindow("Image Window", 1200, 800)
        cv.imshow("Image Window", img)
        cv.waitKey(0)  
        cv.destroyAllWindows()
    return cubesList, pairs
    
def redoRot(poseList, rot):
    for i in range(len(poseList)):
        poseList[i][:3,:3] = rot
    return poseList

def solveA(robot, camera):
    start(robot)
    moveBase(robot, 30)
    cubesList, _ = locateAllCubes(camera)
    moveBase(robot, -30)
    cubes = cubesList[0]
    q = robot.get_q()
    homePose = robot.fk(q)
    # print(cubes)
    # print()
    avgz = sum(mat[2, 3] for mat in cubes)/len(cubes)

    for i, cubePose in enumerate(cubes):
        gripperOpen(robot)
        cubePose[2,3] = avgz + 0.06 #TEST
        linMoveCubeOrHole(robot, cubePose, pick=True)
        moveCubeOrHolePose(robot, homePose)
        moveBase(robot, 30+i*5)
        moveGripperXYZ(robot, 0., 0., -0.3)
        gripperOpen(robot)
        moveCubeOrHolePose(robot, homePose)

    # moveToPose(robot, cubePose)

def solveB(robot, camera):
    start(robot)
    q0 = robot.get_q()
    pose_home = robot.fk(q0)
    curRot = pose_home[:3,:3]

    moveBase(robot, 30)
    cubesList, aruco_pairs = locateAllCubes(camera)
    moveBase(robot, -30)
    
    cubesA, cubesB = cubesList[0], cubesList[1]
    cubesA = redoRot(cubesA, curRot) # because planar solution
    cubesB = redoRot(cubesB, curRot)
    # print("A: ", cubesA)
    # print("B: ", cubesB)
    answer = input(f"Cubes are located on board A: {aruco_pairs[0]['ids']} or B: {aruco_pairs[1]['ids']}").lower()
    # print(cubesA)
    # print()
    # print(cubesB)

    if answer == "a":
        print("You chose A.")
        moveCubesToHoles(robot, cubesA, cubesB)
    elif answer == "b":
        print("You chose B.")
        moveCubesToHoles(robot, cubesB, cubesA)
    else:
        print("Invalid input, please answer with 'A' or 'B'.")


def moveCubesToHoles(robot, cubesPoses, holesPoses):
    q = robot.get_q()
    homePose = robot.fk(q)

    avgz_cubes = sum(mat[2, 3] for mat in cubesPoses)/len(cubesPoses) + 0.055
    avgz_holes = sum(mat[2, 3] for mat in holesPoses)/len(holesPoses) + 0.06
    print("avgz: ", avgz_cubes, avgz_holes)

    for i, (cubePose, holePose) in enumerate(zip(cubesPoses, holesPoses)):
        gripperOpen(robot)
        cubePose[2,3] = avgz_cubes  #TEST
        linMoveCubeOrHole(robot, cubePose, pick=True)
        moveCubeOrHolePose(robot, homePose)
        holePose[2,3] = avgz_holes 
        linMoveCubeOrHole(robot, holePose, pick=False)
        moveCubeOrHolePose(robot, homePose)


def linMoveCubeOrHole(robot, cubePose, pick):

    listOfPoses = []
    start_above = 0.03
    step = 0.006
    n = int(start_above/step)
    z_values = np.linspace(start_above, step, n)
    curOff = np.array([[0, 0, z] for z in z_values])

    R_base2cube = cubePose[:3,:3]
    t_base2cube = cubePose[:3, 3]
    for i in range(len(curOff)):
        t_offset = t_base2cube - (R_base2cube @ curOff[i]).flatten()
        T_base2spaceAboveCube = np.eye(4)
        T_base2spaceAboveCube[:3,:3] = R_base2cube
        T_base2spaceAboveCube[:3, 3] = t_offset
        listOfPoses.append(T_base2spaceAboveCube)
    
    listOfPoses.append(cubePose)
    
    for pose in listOfPoses:
        moveCubeOrHolePose(robot, pose)
    if pick:
        gripperGrab(robot)
    else: 
        gripperOpen(robot)
    time.sleep(2)
    listOfPoses.reverse()
    listOfPoses.pop(0)
    for pose in listOfPoses:
        moveCubeOrHolePose(robot, pose)
    

def moveCubeOrHolePose(robot, pose):
    best_q = validIkForCubesOrHoles(robot, pose)
    if best_q is None:
        return 0
    robot.move_to_q(best_q)
    robot.wait_for_motion_stop()

def test(robot):
    q = robot.get_q()
    pose = robot.fk(q)
    Rx = np.array([[1, 0, 0],
                [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
    
    Ry = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
                [0, 1, 0],
                [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])

    Rz = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0],
                [np.sin(np.pi/2), np.cos(np.pi/2), 0],
                [0, 0, 1]])
    
    R_pose = Rx @ pose[:3,:3]
    pose[:3,:3] = R_pose

    best_q = validIk(robot, pose)
    if best_q is None:
        return 0
    robot.move_to_q(best_q)
    robot.wait_for_motion_stop()
    



def validIkForCubesOrHoles(robot, pose):
    q0 = robot.get_q()
    # Ry = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2)],
    #                 [0, 1, 0],
    #                 [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])
    # Rx = np.array([[1, 0, 0],
    #             [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
    #             [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
    Rz = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0],
                    [np.sin(np.pi/2), np.cos(np.pi/2), 0],
                    [0, 0, 1]])
    valid_q_sol = []
    for i in range(4):
        R_pose = pose[:3,:3]
        R_pose = R_pose @ Rz
        pose[:3,:3] = R_pose
        # print(R_pose)
        possible_q_config = robot.ik(pose)
        valid_q_sol.extend(filterSolOutBounds(robot, possible_q_config))
        

    valid_q_sol = filterSolOutBounds(robot, possible_q_config)
    best_q = None
    if len(valid_q_sol)>0:
        best_q = min(valid_q_sol, key=lambda q: np.linalg.norm(q - q0))
    else:
        print("No solution found for the given x y z")
    return best_q

def printPose(robot):
    q = robot.get_q()
    pose = robot.fk(q)
    print(pose)

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
    # camera.start()
    return camera

def endCamera(camera):
    camera.close()

def resetCamera(camera):
    endCamera(camera)
    cam = getCamera()
    return cam

def gripperOpen(robot):
    robot.gripper.control_position_relative(0.0)
    robot.wait_for_motion_stop()

def gripperGrab(robot):
    robot.gripper.control_position_relative(0.9)
    robot.wait_for_motion_stop()

functions = {
    "start": start,
    "reset": reset,
    "cage": cage,
    "end": end,
    "moveGripperXYZ": moveGripperXYZ,
    # "setGripperDown": setGripperDown,
    "printPose": printPose,
    "moveBase": moveBase, 
    "rotateGripper": rotateGripper,
    "gripperOpen": gripperOpen,
    "solveA": solveA,
    "solveB": solveB,
    "test" : test, 
}

def parse_and_execute(robot, camera, command):
    parts = command.split()
    
    func_name = parts[0]
    if func_name in functions:
        # Prepare arguments based on the function signature
        try: 
            args = []
            for arg in parts[1:]:
                try:
                    # Extract arguments as needed
                    args.append(float(arg))
                except Exception as e:
                    print(f"Error: {e}")
            if func_name == "solveA" or func_name == "solveB" or func_name =="solveC" or func_name=="solveD":
                functions[func_name](robot, camera, *args)   # Call the function with unpacked arguments
            else:
                functions[func_name](robot, *args)   # Call the function with unpacked arguments
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Unknown function: {func_name}")

def monitor_terminal(robot, camera):
    print("Type the function name (e.g., func1, func2) and press Enter to execute.")
    while monitor_terminal_cmds:
        user_input = input("Enter function name: ").strip()
        parse_and_execute(robot, camera, user_input)

if __name__=="__main__":
    # root = os.getcwd()
    # img_dir = os.path.join(root, 'imgs')
    # img_filename = os.path.join(img_dir, f"img0.png")
    # img = cv.imread(img_filename)
    # cubes, _ = locateAllCubes(img)
    # print(cubes)

    robot = CRS97()
    robot.initialize() # performs soft home!!
    camera = getCamera()
    monitor_terminal(robot, camera)