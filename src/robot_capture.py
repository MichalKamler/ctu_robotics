import threading
import numpy as np
import cv2 as cv
import os
from ctu_crs import CRS97
from basler_camera import BaslerCamera
from utils import loadParams, arucoMarkerPoseEstimation

monitor_terminal_cmds = True


camMatrix, distCoeff = loadParams('calibration_ciirc.npz')


# def rotationMatrixToEulerAngles(R):
#     """
#     Convert a rotation matrix R to Euler angles (roll, pitch, yaw).
#     The rotation matrix is assumed to be a 3x3 matrix.
#     """
#     # Pitch (theta) around Y-axis
#     pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
#     # Roll (phi) around X-axis
#     roll = np.arctan2(R[2, 1], R[2, 2])
    
#     # Yaw (psi) around Z-axis
#     yaw = np.arctan2(R[1, 0], R[0, 0])

#     # Convert radians to degrees if needed
#     roll_deg = np.degrees(roll)
#     pitch_deg = np.degrees(pitch)
#     yaw_deg = np.degrees(yaw)

#     return roll_deg, pitch_deg, yaw_deg

# def rotationToMatrix(roll, pitch, yaw):
#     # Roll - around x-axis
#     R_x = np.array([[1, 0, 0],
#                     [0, np.cos(roll), -np.sin(roll)],
#                     [0, np.sin(roll), np.cos(roll)]])
    
#     # Pitch - around y-axis
#     R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                     [0, 1, 0],
#                     [-np.sin(pitch), 0, np.cos(pitch)]])
    
#     # Yaw - around z-axis
#     R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                     [np.sin(yaw), np.cos(yaw), 0],
#                     [0, 0, 1]])
    
#     # Combined rotation matrix
#     return R_z @ R_y @ R_x

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
    print(q)
    print(robot.q_min)
    print(robot.q_max)

def setGripperDown(robot, deg): 
    print("setGripperDown is running")
    q0 = robot.get_q()

    rad = np.deg2rad(deg)
    q = fitSumOfq(q0, [0.0, 0.0, 0.0, 0.0, rad, 0.0])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    print("setGripperDown completed")

def start(robot):
    moveGripperXYZ(robot, 0, 0, -0.2)
    setGripperDown(robot,-90)

def moveGripperXYZ(robot, x, y, z):
    print("moveGripperXYZ is running")
    q0 = robot.get_q()
    pose = robot.fk(q0)

    pose[:3, 3] += np.array([x, y, z])

    best_q = validIk(robot, pose)
    robot.move_to_q(best_q)
    robot.wait_for_motion_stop()
    print("moveGripperXYZ completed")

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
    q = fitSumOfq(q0, [rad, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    print("moveBase completed")

def rotateGripper(robot, deg):
    print("rotateGripper is running")
    rad = np.deg2rad(deg)
    q0 = robot.get_q()
    q = fitSumOfq(q0, [0.0, 0.0, 0.0, 0.0, 0.0, rad])
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    print("rotateGripper completed")

def waitForImg(camera):
    img = camera.grab_image()
    while img is None or img.size<=0:
        print("Image not captured, staying in while loop")
        img = camera.grab_image()
    return img

def captureImgAndPose(robot):
    moveGripperXYZ(robot, 0.13, -0.17, -0.38) # move to the top left corner and minimum z
    x_max = 0.20
    y_max = 0.34
    z_max = 0.12

    x_step = 0.05
    y_step = 0.04
    z_step = 0.04

    x_range = int(x_max/x_step) 
    y_range = int(y_max/y_step) 
    z_range = int(z_max/z_step) 

    counter = 0

    camera: BaslerCamera = BaslerCamera()    
    camera.open()
    camera.set_parameters()
    camera.start()

    rotateGripper(robot, -90) #not sure about the -90 change !!!

    for z in range(z_range):
        for x in range(x_range):
            savePicPose(robot, camera, counter)
            counter += 1
            for y in range(y_range):
                if y == y_range//2:
                    rotateGripper(robot, 180)
                moveGripperXYZ(0, y_step, 0)
                
                savePicPose(robot, camera, counter)
                counter += 1

                setGripperDown(robot, 15)
                savePicPose(robot, camera, counter)
                counter += 1
                setGripperDown(robot, -15)
            
            rotateGripper(robot, -180)
            moveGripperXYZ(x_step, -y_step*y_range, 0)
        moveGripperXYZ(-x_step*x_range, -y_step*y_range, z_step)

def savePicPose(robot, camera, idx):
    root = os.getcwd()
    img_dir = os.path.join(root, 'aruco_calib', 'imgs')
    poses_dir = os.path.join(root, 'aruco_calib', 'poses')
    img_data_dir = os.path.join(root, 'aruco_calib', 'img_data')

    img = waitForImg(camera)
    img_filename = os.path.join(img_dir, f"img{idx}_0.png")
    cv.imwrite(img_filename, img)

    saveQ(robot, idx, poses_dir)

    img_aruco, rvec, tvec = arucoMarkerPoseEstimation(img, cv.aruco.DICT_4X4_50, camMatrix, distCoeff, 8.0, [8.0, 4.0, 0])

    img_filename = os.path.join(img_dir, f"img{idx}_1.png")
    cv.imwrite(img_filename, img_aruco)

    img_data_filename = os.path.join(img_data_dir, f"data{idx}.txt")

    with open(img_data_filename, "w") as file:
        file.write("# rvecs: \n")
        np.savetxt(file, [rvec], fmt='%s')

        file.write("# tvecs: \n")
        np.savetxt(file, [tvec], fmt='%s')

    #TODO save img data to txt and processed img for control

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
    monitor_terminal(robot)
