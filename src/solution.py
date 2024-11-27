import threading
import numpy as np
from enum import Enum
import time
import cv2 as cv
import os
from ctu_crs import CRS97
from basler_camera import BaslerCamera
from utils import loadCamDist, arucoMarkersFinder, pairUpAruco, locateCenterOfCubes, drawFoundCubes, loadRT
import simple_control


camMatrix, distCoeff = loadCamDist('npz/calibration_ciirc.npz')
R_base2cam, t_base2cam = loadRT('npz/R_t_base2cam.npz')

T_base2cam = np.eye(4)
T_base2cam[:3, :3] = R_base2cam
T_base2cam[:3, 3] = t_base2cam.flatten()
monitor_terminal_cmds = True


def locateAllCubes(img):
    img, allT_base2marker, ids = arucoMarkersFinder(img, camMatrix, distCoeff, 0.036)
    pairs = pairUpAruco(allT_base2marker, ids)
    cubes = locateCenterOfCubes(pairs[0])
    img = drawFoundCubes(img, camMatrix, distCoeff, cubes, T_base2cam)


    if img is None:
        print("Error: Could not load image.")
    else:
        cv.namedWindow("Image Window", cv.WINDOW_NORMAL)
        cv.resizeWindow("Image Window", 600, 400)
        cv.imshow("Image Window", img)
        cv.waitKey(0)  
        cv.destroyAllWindows()
    




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

if __name__=="__main__":
    root = os.getcwd()
    img_dir = os.path.join(root, 'imgs')
    img_filename = os.path.join(img_dir, f"img0.png")
    img = cv.imread(img_filename)
    locateAllCubes(img)

    robot = CRS97()
    robot.initialize() # performs soft home!!
    # print(camMatrix, distCoeff)
    monitor_terminal(robot)