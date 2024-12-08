import cv2 as cv
import numpy as np
import os
from scipy.optimize import minimize
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import loadCamDist
camMatrix, distCoeff = loadCamDist('npz/calibration_ciirc.npz')

def loadPose(i, directory):
    """
    Load the pose (4x4 matrix) from a saved file.

    Parameters:
        i (int): The index of the saved file (matches the `output{i}.txt` format).
        directory (str): The directory where the file is saved. Defaults to "aruco_calib/poses".

    Returns:
        np.ndarray: The 4x4 pose matrix.
    """
    if directory is None:
        directory = "aruco_calib/poses"
    
    filename = os.path.join(directory, f"output{i}.txt")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")

    # Read the file and extract the pose section
    pose_start = False
    pose_lines = []
    
    with open(filename, "r") as file:
        for line in file:
            # Look for the Pose section header
            if line.startswith("# Pose (4x4 matrix):"):
                pose_start = True
                continue
            # Stop reading once another section starts
            elif line.startswith("#") and pose_start:
                break
            # Collect pose lines
            elif pose_start and line.strip():
                pose_lines.append(line.strip())
    
    # Convert the collected pose lines to a NumPy array
    if pose_lines:
        pose = np.array([list(map(float, line.split())) for line in pose_lines])
        if pose.shape == (4, 4):
            return pose
        else:
            raise ValueError("Pose matrix in the file is not 4x4.")
    else:
        raise ValueError("Pose matrix section not found in the file.")

def loadAllA(n):
    A_list = [] 
    src = os.path.dirname(os.getcwd())
    pose_path = os.path.join(src, 'aruco_calib/poses')
    for i in range(n):
        A_list.append(loadPose(i, pose_path))
    return A_list

def loadRvecsTvecsUi(idx, directory):
    """
    Load the rvecs and tvecs from a saved file.

    Parameters:
        idx (int): The index of the saved file (matches the `data{idx}.txt` format).
        directory (str): The directory where the file is saved. Defaults to "aruco_calib/data".

    Returns:
        tuple: A tuple containing two np.ndarrays:
            - rvec (1x3): The rotation vector.
            - tvec (1x3): The translation vector.
    """
    
    filename = os.path.join(directory, f"data{idx}.txt")
    
    if not os.path.exists(filename):
        return None, None, None
        # raise FileNotFoundError(f"File {filename} does not exist.")
    
    # Read the file and extract rvecs and tvecs sections
    rvec_start = False
    tvec_start = False
    rvec_lines = []
    tvec_lines = []
    ui_lines = []
    
    with open(filename, "r") as file:
        for line in file:
            # Identify the start of rvecs section
            if line.startswith("# rvecs:"):
                rvec_start = True
                tvec_start = False
                ui_start = False
                continue
            # Identify the start of tvecs section
            elif line.startswith("# tvecs:"):
                rvec_start = False
                tvec_start = True
                ui_start = False
                continue
            elif line.startswith("# ui_list:"):
                rvec_start = False
                tvec_start = False
                ui_start = True
                continue
            
            # Accumulate rvec data
            if rvec_start and line.strip():
                values = line.strip().split()
                if len(values) == 3:  # Ensure it has 3 values
                    rvec_lines.append([float(v) for v in values])
            
            # Accumulate tvec data
            if tvec_start and line.strip():
                values = line.strip().split()
                if len(values) == 3:  # Ensure it has 3 values
                    tvec_lines.append([float(v) for v in values])
            
            # Accumulate ui data
            if ui_start and line.strip():
                values = line.strip().split()
                if len(values) == 2:  # Ensure it has 3 values
                    ui_lines.append([float(v) for v in values])

    
    # Convert the accumulated lines to NumPy arrays
    rvec = np.array(rvec_lines)
    tvec = np.array(tvec_lines)
    ui = np.array(ui_lines)

    # Validate that rvec and tvec were successfully loaded
    if rvec.size != 3 or tvec.size != 3:
        raise ValueError(f"rvec or tvec data in file {filename} is not valid. Expected 3 elements each.")

    return rvec, tvec, ui

def loadAllB(n):
    B_list = []
    ui_list = []
    src = os.path.dirname(os.getcwd())
    rvecs_tvecs_path = os.path.join(src, 'aruco_calib/img_data')
    for i in range(n):
        rvec, tvec, ui = loadRvecsTvecsUi(i, rvecs_tvecs_path)
        if rvec is None or tvec is None:
            B_list.append(None)
            continue
        B = np.eye(4)
        rotMat, _ = cv.Rodrigues(rvec)
        B[:3, :3] = rotMat
        B[:3, 3] = tvec
        B_list.append(B)
        ui_list.append(ui)
    return B_list, ui_list

def invert_homogeneous_transform(T):
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix.

    Parameters:
        T (numpy.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        numpy.ndarray: The inverse of the input homogeneous transformation matrix.
    """
    # Extract rotation (R) and translation (t)
    R = T[:3, :3]
    t = T[:3, 3]
    
    # Compute the inverse
    R_inv = R.T  # Transpose of the rotation matrix
    t_inv = -R_inv @ t  # Apply the inverse rotation to the negative translation
    
    # Construct the inverse homogeneous matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    
    return T_inv


def matToParam(T1, T2):
    # R1 = T1[:3,:3]
    t1 = T1[:3, 3]
    # R2 = T2[:3,:3]
    t2 = T2[:3, 3]
    return np.concatenate([t1.flatten(), t2.flatten()])
    # return np.concatenate([R1.flatten(), t1.flatten(), R2.flatten(), t2.flatten()])
    
def paramToMat(param, T1, T2):
    # R1 = param[:9].reshape(3,3)
    # t1 = param[9:12]
    # R2 = param[12:21].reshape(3,3)
    # t2 = param[21:]
    # T1 = np.eye(4)
    # T1[:3,:3] = R1
    # T1[:3, 3] = t1
    # T2 = np.eye(4)
    # T2[:3,:3] = R2
    # T2[:3, 3] = t2
    T1[:3, 3] = param[:3]
    T2[:3, 3] = param[3:]
    return T1, T2

def projection2d(T_gripper2target, T_base2camera, T_base2gripper):
    T_camera2base = invert_homogeneous_transform(T_base2camera)
    T_cam2target = T_camera2base @ T_base2gripper @ T_gripper2target
    
    X = T_cam2target[:3, 3]
    image_point, _ = cv.projectPoints(X, rvec=np.zeros(3), tvec=np.zeros(3), cameraMatrix=camMatrix, distCoeffs=distCoeff)
    pix_x, pix_y = image_point[0][0]
    return pix_x, pix_y
    

def reprojectionError(params, T_gripper2target, T_base2camera, T_base2gripper_list, ui):
    T_gripper2target, T_base2camera = paramToMat(param=params, T1=T_gripper2target, T2=T_base2camera)
    error = 0.0

    for i, T_base2gripper in enumerate(T_base2gripper_list): 
        projection_x, projection_y = projection2d(T_gripper2target, T_base2camera, T_base2gripper)
        error += abs(ui[i][0]-projection_x) + abs(ui[i][1]-projection_y)
    return error



if __name__=="__main__":
    n = 80

    A_list = loadAllA(n)
    B_list, ui_list = loadAllB(n)

    idx_banned = []
    for i in range(len(B_list)):
        if B_list[i] is None:
            idx_banned.append(i)
    
    T_base2gripper = [value for idx, value in enumerate(A_list) if idx not in idx_banned]
    T_cam2target = [value for idx, value in enumerate(B_list) if idx not in idx_banned] #recorded points in cam coord sys

    R_base2gripper = [T[:3,:3] for T in T_base2gripper]
    t_base2gripper = [T[:3, 3] for T in T_base2gripper]
    R_cam2target= [T[:3,:3] for T in T_cam2target] #due to my already did translation is same as cam2gripper
    t_cam2target = [T[:3, 3] for T in T_cam2target]


    R_gripper2target, t_gripper2target, R_base2camera, t_base2camera = cv.calibrateRobotWorldHandEye(R_base2gripper, t_base2gripper, R_cam2target, t_cam2target, method=cv.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
    # Save Calibration Parameters
    src = os.path.dirname(os.getcwd()) #gets parent working dir
    paramPath = os.path.join(src, 'npz/R_t_base2cam.npz')
    print(R_base2camera, t_base2camera)
    print(R_gripper2target, t_gripper2target)
    np.savez(paramPath, R=R_base2camera, t=t_base2camera)


    T_gripper2target = np.eye(4)
    T_gripper2target[:3,:3] = R_gripper2target
    T_gripper2target[:3, 3] = t_gripper2target.flatten()

    T_base2camera = np.eye(4)
    T_base2camera[:3,:3] = R_base2camera
    T_base2camera[:3, 3] = t_base2camera.flatten()

    print("T_base2camera: \n", T_base2camera)

    ui = np.array([point.flatten() for point in ui_list])  # List of 2D points

    initial_guess_flat = matToParam(T_gripper2target, T_base2camera)

    print("Error before optimization: ", reprojectionError(initial_guess_flat, T_gripper2target, T_base2camera, T_base2gripper, ui))

    result = minimize(reprojectionError, initial_guess_flat, args=(T_gripper2target, T_base2camera, T_base2gripper, ui))

    optimized_T_gripper2target, optimized_T_base2camera = paramToMat(result.x, T_gripper2target, T_base2camera)


    print(optimized_T_gripper2target)
    print(optimized_T_base2camera)

    final_guess_flat = matToParam(optimized_T_gripper2target, optimized_T_base2camera)

    print("Error after optimization: ", reprojectionError(final_guess_flat, T_gripper2target, T_base2camera, T_base2gripper, ui))

    print("optimized_T_base2camera: \n", optimized_T_base2camera)

    opt_R_base2camera = optimized_T_base2camera[:3,:3]
    opt_t_base2camera = optimized_T_base2camera[:3, 3]
    paramPath = os.path.join(src, 'npz/R_t_base2cam.npz')
    np.savez(paramPath, R=opt_R_base2camera, t=opt_t_base2camera)


    



