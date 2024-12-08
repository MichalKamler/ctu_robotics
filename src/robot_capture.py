import threading
import numpy as np
from enum import Enum
import time
import cv2 as cv
import os
from ctu_crs import CRS97
from basler_camera import BaslerCamera
from utils import loadCamDist, arucoMarkerPoseEstimation

monitor_terminal_cmds = True

class ArucoType(Enum):
    DICT_4X4_50 = cv.aruco.DICT_4X4_50
    DICT_4X4_100 = cv.aruco.DICT_4X4_100
    DICT_4X4_250 = cv.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv.aruco.DICT_4X4_1000


camMatrix, distCoeff = loadCamDist('npz/calibration_ciirc.npz')

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

def help(robot):
    print("reset - resets motors of the robot")
    print("cage - waits for the robot to stop moving and releases the robot so you can safely enter the cage")
    print("end - moves robot to home and terminates the connection")
    # "moveGpripperXYZ": moveGpripperXYZ,
    # "setGripperDown": setGripperDown,
    # "printPose": printPose,
    # "moveBase" : moveBase, 

def printPose(robot):
    q = robot.get_q()
    pose = robot.fk(q)
    print(pose)
    # print(q)
    # print(robot.q_min)
    # print(robot.q_max)

def setGripperDown(robot, deg): 
    # print("setGripperDown is running")
    q0 = robot.get_q()

    rad = np.deg2rad(deg)
    q = fitSumOfq(robot, q0, [0.0, 0.0, 0.0, 0.0, rad, 0.0])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    # print("setGripperDown completed")

def start(robot):
    q0 = robot.get_q()
    q = fitSumOfq(robot, q0, [0.0, 0.0, -np.pi/4, 0.0, -np.pi/4, 0.0])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    # moveGripperXYZ(robot, 0, 0, -0.2)
    # setGripperDown(robot,-90)

def moveGripperXYZ(robot, x, y, z):
    # print("moveGripperXYZ is running")
    q0 = robot.get_q()
    pose = robot.fk(q0)

    pose[:3, 3] += np.array([x, y, z])

    best_q = validIk(robot, pose)
    if best_q is None:
        print(f"failed to move x: {x}, y: {y}, z: {z}")
        return 0
    robot.move_to_q(best_q)
    robot.wait_for_motion_stop()
    # print("moveGripperXYZ completed")
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
    # print("rotateGripper is running")
    rad = np.deg2rad(deg)
    q0 = robot.get_q()
    q = fitSumOfq(robot, q0, [0.0, 0.0, 0.0, 0.0, 0.0, rad])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    # print("rotateGripper completed")

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

def captureImgAndPose(robot):
    if not moveGripperXYZ(robot, 0.00, -0.22, -0.35): # move to the top left corner and minimum z
        print("nejde nic")
        return
    x_max = 0.42
    y_max = 0.40
    z_max = 0.16

    x_step = 0.06 #trust bro some mistake is integrating so this nees to be larger :(
    y_step = 0.04
    z_step = 0.04

    # x_step = 0.14
    # y_step = 0.22
    # z_step = 0.03

    x_range = int(x_max/x_step) 
    y_range = int(y_max/y_step) 
    z_range = int(z_max/z_step) 

    print(x_range, y_range, z_range)

    counter = 0

    camera = getCamera()
    move_y_times = 0

    rotateGripper(robot, -90) #not sure about the -90 change !!!

    q0 = robot.get_q()
    pose_home = robot.fk(q0)
    curRot = pose_home[:3,:3]

    for z in range(z_range):
        movePose(robot, pose_home)
        moveGripperXYZ(robot, 0., 0., z*z_step)
        for x in range(x_range):
            resetRot(robot,curRot)
            savePicPose(robot, camera, counter)
            counter += 1
            for y in range(y_range):
                if y == y_range//2:
                    rotateGripper(robot, 180)
                if not moveGripperXYZ(robot, 0, y_step + y_step * move_y_times, 0): #no possible config, skip this position
                    move_y_times += 1
                    continue
                else:
                    move_y_times = 0

                
                if not savePicPose(robot, camera, counter):
                    camera = resetCamera(camera)
                counter += 1

                setGripperDown(robot, 30)
                savePicPose(robot, camera, counter)
                counter += 1
                setGripperDown(robot, -30)
            
            rotateGripper(robot, -180)
            moveGripperXYZ(robot, 0, -y_step*y_range, 0)
            resetRot(robot,curRot)
            moveGripperXYZ(robot, x_step, 0, 0)            
        # moveGripperXYZ(robot, -x_step*x_range, 0, 0)
        # moveGripperXYZ(robot, 0, 0, z_step)

def movePose(robot, pose):
    best_q = validIk(robot, pose)
    if best_q is None:
        return 0
    robot.move_to_q(best_q)
    robot.wait_for_motion_stop()

def resetRot(robot, rot):
    q0 = robot.get_q()
    pose = robot.fk(q0)

    pose[:3,:3] = rot

    best_q = validIk(robot, pose)
    if best_q is None:
        return 0
    robot.move_to_q(best_q)
    robot.wait_for_motion_stop()


def savePicPose(robot, camera, idx):
    root = os.getcwd()
    
    img_dir = os.path.join(root, 'aruco_calib', 'imgs')
    poses_dir = os.path.join(root, 'aruco_calib', 'poses')
    img_data_dir = os.path.join(root, 'aruco_calib', 'img_data')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(img_data_dir, exist_ok=True)

    time.sleep(0.1) #to make sure that the img will not be blurry
    
    img = waitForImg(camera)
    if img is None:
        #skip this position
        return False #fail
    img_filename = os.path.join(img_dir, f"img{idx}_0.png")
    cv.imwrite(img_filename, img)

    saveQ(robot, idx, poses_dir)

    img_aruco, rvec, tvec, _ = arucoMarkerPoseEstimation(img, ArucoType.DICT_4X4_50, camMatrix, distCoeff, 6.0)

    if rvec is None or tvec is None:
        return True

    img_filename = os.path.join(img_dir, f"img{idx}_1.png")
    cv.imwrite(img_filename, img_aruco)

    img_data_filename = os.path.join(img_data_dir, f"data{idx}.txt")

    with open(img_data_filename, "w") as file:
        file.write("# rvecs: \n")
        np.savetxt(file, rvec, fmt='%s')

        file.write("# tvecs: \n")
        np.savetxt(file, tvec, fmt='%s')

    return True #success

def saveQ(robot, i, directory =None):
    # print("saveQ is running")
    if directory is None: 
        directory = "aruco_calib/poses"
    
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"output{i}.txt")


    q = robot.get_q() # ndarray
    pose = robot.fk(q) # 4x4 array
    xyz = robot.fk_flange_pos(q) #ndarray

    with open(filename, "w") as file: 
        file.write("# q: \n")
        np.savetxt(file, [q], fmt='%s')

        file.write("\n# Pose (4x4 matrix): \n")
        np.savetxt(file, pose, fmt='%.6f')

        file.write("\n# XYZ: \n")
        np.savetxt(file, [xyz], fmt='%s')
    # print("saveQ completed")

def gripperOpen(robot):
    robot.gripper.control_position_relative(0.0)
    robot.wait_for_motion_stop()

def gripperGrab(robot):
    robot.gripper.control_position_relative(0.99)
    robot.wait_for_motion_stop()

def end(robot):
    print("end is running!")
    robot.soft_home()
    robot.close()
    monitor_terminal_cmds = False
    print("end completed!")

functions = {
    "help": help,
    "start": start,
    "moveGripperXYZ": moveGripperXYZ,
    "setGripperDown": setGripperDown,
    "printPose": printPose,
    "moveBase": moveBase, 
    "rotateGripper": rotateGripper,
    "saveQ": saveQ,
    "reset": reset,
    "cage": cage,
    "gripperOpen": gripperOpen,
    "gripperGrab": gripperGrab,
    "captureImgAndPose": captureImgAndPose,
    "end": end,
}

def parse_and_execute(robot, command):
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
            functions[func_name](robot, *args)  # Call the function with unpacked arguments
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Unknown function: {func_name}")

def monitor_terminal(robot):
    print("Type the function name (e.g., func1, func2) and press Enter to execute.")
    while monitor_terminal_cmds:
        user_input = input("Enter function name: ").strip()
        parse_and_execute(robot, user_input)


# robot.fk_flange_pos(q: ArrayLike) -> x y z of the gripper with regards to the base 0 0 0

# robot.get_q(self) -> np.ndarray -> Get current joint configuration.

# robot.move_to_q(q: ArrayLike)

# robot.fk(self, q: ArrayLike) -> np.ndarray: -> pose represented as 4x4 matrix SE3 wrt base 

# robot.ik(self, pose: np.ndarray) -> list[np.ndarray]:


if __name__ == "__main__":
    robot = CRS97()
    robot.initialize() # performs soft home!!
    # print(camMatrix, distCoeff)
    monitor_terminal(robot)
