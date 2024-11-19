import cv2 as cv
import numpy as np
import os

def loadPose(i, directory=None):
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
            elif pose_start:
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
    for i in range(n):
        A_list.append(loadPose(i))
    return A_list

def loadRvecsTvecs(idx, directory=None):
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
    if directory is None:
        directory = "aruco_calib/data"
    
    filename = os.path.join(directory, f"data{idx}.txt")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")
    
    # Read the file and extract rvecs and tvecs sections
    rvec_start = False
    tvec_start = False
    rvec = None
    tvec = None
    
    with open(filename, "r") as file:
        for line in file:
            # Identify the start of rvecs section
            if line.startswith("# rvecs:"):
                rvec_start = True
                continue
            # Identify the start of tvecs section
            elif line.startswith("# tvecs:"):
                rvec_start = False
                tvec_start = True
                continue
            
            # Read rvec data
            if rvec_start:
                rvec = np.fromstring(line.strip(), sep=' ')
                rvec_start = False  # Only a single line for rvec
            
            # Read tvec data
            if tvec_start:
                tvec = np.fromstring(line.strip(), sep=' ')
                tvec_start = False  # Only a single line for tvec

    # Validate that rvec and tvec were successfully loaded
    if rvec is None or tvec is None:
        raise ValueError(f"rvec or tvec data not found in file {filename}.")
    
    return rvec, tvec

def loadAllB(n):
    B_list = []
    for i in range(n):
        rvec, tvec = loadRvecsTvecs(i)
        B = np.eye(4)
        rotMat, _ = cv.Rodrigues(rvec)
        B[:3, :3] = rotMat
        B[:3, 3] = tvec
        B_list.append(B)
    return B_list

if __name__=="__main__":
    n = 10 

    A_list = loadAllA(n)
    B_list = loadAllB(n)

    R_gripper2base = [A[:3,:3] for A in A_list]
    t_gripper2base = [A[:3, 3] for A in A_list]
    R_target2cam = [B[:3,:3] for B in B_list]
    t_target2cam = [B[:3, 3] for B in B_list]

    R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

    # Save Calibration Parameters
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'R_t_cam2gripper.npz')
    np.savez(paramPath, R=R_cam2gripper, t=t_cam2gripper)
