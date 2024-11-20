import os
import numpy as np
from enum import Enum
import cv2 as cv

class ArucoType(Enum):
    DICT_4X4_50 = cv.aruco.DICT_4X4_50
    DICT_4X4_100 = cv.aruco.DICT_4X4_100
    DICT_4X4_250 = cv.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv.aruco.DICT_4X4_1000


def loadParams(name):
    curFolder = os.path.dirname(os.path.abspath(__file__))
    # Load Calibration Parameters
    paramPath = os.path.join(curFolder, name)

    # Load the .npz file
    params = np.load(paramPath)

    # Extract the parameters
    repError = params['repError']
    camMatrix = params['camMatrix']
    distCoeff = params['distCoeff']
    rvecs = params['rvecs']
    tvecs = params['tvecs']

    # Now you can use these parameters for camera calibration or further processing
    print(f"Calibration parameters loaded from {paramPath}")
    return camMatrix, distCoeff

def rotationToMatrix(roll, pitch, yaw):
    # Roll - around x-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    # Pitch - around y-axis
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Yaw - around z-axis
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    return R_z @ R_y @ R_x

def arucoMarkerPoseEstimation(img, aruco_type, camera_matrix, dist_coeffs, aruco_side_size, gripper_offest): 

    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type.value) 
    aruco_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict,aruco_params)

    world_points = np.array([[0.,0.,0.], # top left
                                [aruco_side_size,0.,0.], # top right
                                [aruco_side_size,aruco_side_size,0.], # bottom right
                                [0.,aruco_side_size,0.]  # bottom left
    ])


    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(img_gray)
    rvecs_transformed = None
    vector_new = None
    if ids is not None: 
        img = cv.aruco.drawDetectedMarkers(img,corners,ids)

        for corner in corners: 

            center_x = int(corner[0][:, 0].mean())
            center_y = int(corner[0][:, 1].mean())

            success,rvecs,tvecs = cv.solvePnP(world_points,corner,camera_matrix,dist_coeffs)
            if not success:
                continue

            rotation_matrix, _ = cv.Rodrigues(rvecs)

            flip_matrix = np.array([
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]
                            ], dtype=np.float32)
            transformed_rotation_matrix = rotation_matrix @ flip_matrix
            euler_angles = cv.decomposeProjectionMatrix(np.hstack((transformed_rotation_matrix, tvecs)))[6]
                    
            tilt_info = f"Center: ({center_x}, {center_y}), Tilt (Roll, Pitch, Yaw): {euler_angles[0][0]:.2f}, {euler_angles[1][0]:.2f}, {euler_angles[2][0]:.2f}"
            # print(tilt_info)
            rvecs_transformed, _ = cv.Rodrigues(transformed_rotation_matrix)

            R = rotationToMatrix(np.deg2rad(euler_angles[0][0]), np.deg2rad(euler_angles[1][0]), np.deg2rad(euler_angles[2][0]))
            R_inv = R.T
            vector_local = R_inv @ tvecs

            vector_local += np.array([[gripper_offest[0]], [gripper_offest[1]], [gripper_offest[2]]])
            vector_new = R @ vector_local


            # print(tvecs)
            cv.putText(img, tilt_info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            
            # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs_transformed, tvecs, 1) 
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs_transformed, vector_new, 1)


    return img, rvecs_transformed, vector_new