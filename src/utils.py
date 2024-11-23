import os
import numpy as np
from enum import Enum
import cv2 as cv

class ArucoType(Enum):
    DICT_4X4_50 = cv.aruco.DICT_4X4_50
    DICT_4X4_100 = cv.aruco.DICT_4X4_100
    DICT_4X4_250 = cv.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv.aruco.DICT_4X4_1000


def loadParams(dirname):
    curFolder = os.path.dirname(os.path.abspath(__file__))
    # Load Calibration Parameters
    paramPath = os.path.join(curFolder, dirname)

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

    world_points = aruco_side_size * np.array([[0.,0.,0.], # top left
                                [1.,0.,0.], # top right
                                [1.,1.,0.], # bottom right
                                [0.,1.,0.]  # bottom left
    ])


    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(img_gray)
    rvecs_transformed = None
    vector_new = None
    if ids is not None: 
        img = cv.aruco.drawDetectedMarkers(img,corners,ids)

        for corner in corners: 

            success,rvecs,tvecs = cv.solvePnP(world_points,corner,camera_matrix,dist_coeffs)
            if not success:
                continue

            rotation_matrix, _ = cv.Rodrigues(rvecs)

            flip_matrix = np.array([
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]
                            ], dtype=np.float32)
            
            theta = np.radians(90)  # Rotation angle in degrees
            c = np.cos(theta)
            s = np.sin(theta)
            rotation_y = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])

            transformed_rotation_matrix = rotation_matrix @ flip_matrix
            euler_angles = cv.decomposeProjectionMatrix(np.hstack((transformed_rotation_matrix, tvecs)))[6]
            transformed_rotation_matrix = transformed_rotation_matrix @ rotation_y
            rvecs_transformed, _ = cv.Rodrigues(transformed_rotation_matrix)

            R = rotationToMatrix(np.deg2rad(euler_angles[0][0]), np.deg2rad(euler_angles[1][0]), np.deg2rad(euler_angles[2][0]))
            R_inv = R.T
            vector_local = R_inv @ tvecs

            vector_local += np.array([[gripper_offest[0]], [gripper_offest[1]], [gripper_offest[2]]])
            vector_new = R @ vector_local

            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs_transformed, vector_new, 1)


    return img, rvecs_transformed, vector_new

def arucoMarkersFinder(img, aruco_type, camera_matrix, dist_coeffs, aruco_side_size): 

    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type.value) 
    aruco_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict,aruco_params)

    world_points = aruco_side_size * np.array([[0.,0.,0.], # top left
                                [1.,0.,0.], # top right
                                [1.,1.,0.], # bottom right
                                [0.,1.,0.]  # bottom left
    ])

    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(img_gray)
    rvecs_transformed = None
    vector_new = None
    if ids is not None: 
        img = cv.aruco.drawDetectedMarkers(img,corners,ids)

        for corner in corners: 

            success,rvecs,tvecs = cv.solvePnP(world_points,corner,camera_matrix,dist_coeffs)
            if not success:
                continue

            rotation_matrix, _ = cv.Rodrigues(rvecs)

            flip_matrix = np.array([
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]
                            ], dtype=np.float32)
            
            theta = np.radians(90)  # Rotation angle in degrees
            c = np.cos(theta)
            s = np.sin(theta)
            rotation_y = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])

            transformed_rotation_matrix = rotation_matrix @ flip_matrix
            euler_angles = cv.decomposeProjectionMatrix(np.hstack((transformed_rotation_matrix, tvecs)))[6]
            transformed_rotation_matrix = transformed_rotation_matrix @ rotation_y
            # tilt_info = f"Center: ({center_x}, {center_y}), Tilt (Roll, Pitch, Yaw): {euler_angles[0][0]:.2f}, {euler_angles[1][0]:.2f}, {euler_angles[2][0]:.2f}"
            rvecs_transformed, _ = cv.Rodrigues(transformed_rotation_matrix)

            R = rotationToMatrix(np.deg2rad(euler_angles[0][0]), np.deg2rad(euler_angles[1][0]), np.deg2rad(euler_angles[2][0]))
            R_inv = R.T
            vector_local = R_inv @ tvecs

            vector_local += np.array([[aruco_side_size/2], [aruco_side_size/2], [0.0]])
            vector_new = R @ vector_local
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs_transformed, vector_new, 0.0)

            print(ids, rvecs, vector_new)


    return img, rvecs_transformed, vector_new