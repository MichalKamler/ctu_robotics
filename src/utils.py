import os
import numpy as np
from enum import Enum
import cv2 as cv
import csv

class ArucoType(Enum):
    DICT_4X4_50 = cv.aruco.DICT_4X4_50
    DICT_4X4_100 = cv.aruco.DICT_4X4_100
    DICT_4X4_250 = cv.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv.aruco.DICT_4X4_1000

def loadCamDist(dirname):
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
    # print(f"Calibration parameters loaded from {paramPath}")
    return camMatrix, distCoeff

def loadRT(dirname):
    curFolder = os.path.dirname(os.path.abspath(__file__))
    # Load Calibration Parameters
    paramPath = os.path.join(curFolder, dirname)

    # Load the .npz file
    params = np.load(paramPath)

    # Extract the parameters
    R = params['R']
    t = params['t']

    # Now you can use these parameters for camera calibration or further processing
    # print(f"Calibration parameters loaded from {paramPath}")
    return R, t

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

def constRotMatrix(mat):
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

    return mat @ flip_matrix @ rotation_y
    
def arucoMarkerPoseEstimation(img, aruco_type, camera_matrix, dist_coeffs, aruco_side_size): 

    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_type.value) 
    aruco_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict,aruco_params)

    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(img_gray)

    rvecs, tvecs = None, None
    ui_list = []
    if ids is not None: 
        img = cv.aruco.drawDetectedMarkers(img,corners,ids)

        for corner in corners: 
            center = np.mean(corner[0], axis=0)  
            ui_list.append(center) 

            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers([corner], markerLength=aruco_side_size, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], aruco_side_size)
    if rvecs is not None and tvecs is not None:
        return img, rvecs[0], tvecs[0], ui_list
    else: 
        return img, None, None, []

def arucoMarkersFinder(img, camera_matrix, dist_coeffs, aruco_side_size): 

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict,aruco_params)

    R_base2cam, t_base2cam = loadRT('npz/R_t_base2cam.npz')

    T_base2cam = np.eye(4)
    T_base2cam[:3, :3] = R_base2cam
    T_base2cam[:3, 3] = t_base2cam.flatten()

    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(img_gray)

    allT_base2marker = []

    if ids is not None: 
        img = cv.aruco.drawDetectedMarkers(img,corners,ids)

        for corner in corners: 
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers([corner], markerLength=aruco_side_size, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
            
            t_camera2marker = tvecs[0].ravel()
            R_camera2marker, _ = cv.Rodrigues(rvecs[0])

            R_camera2marker = constRotMatrix(R_camera2marker)
            
            T_camera2marker = np.eye(4)
            T_camera2marker[:3, :3] = R_camera2marker
            T_camera2marker[:3, 3] = t_camera2marker

            T_base2marker = T_base2cam @ T_camera2marker

            allT_base2marker.append(T_base2marker)

            rvecs_transformed = cv.Rodrigues(R_camera2marker)
            cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs_transformed[0], tvecs[0], aruco_side_size)


    return img, allT_base2marker, ids

def pairUpAruco(allT_base2marker, ids):
    pairedArucoMarkers = []
    ids, allT_base2marker = zip(*sorted(zip(ids, allT_base2marker)))
    for i in range(0, len(ids), 2):
        if i + 1 < len(ids):  # Ensure there's a pair
            pair = {
                'ids': (ids[i][0], ids[i + 1][0]),
                'T': (allT_base2marker[i], allT_base2marker[i + 1]),
            }
            pairedArucoMarkers.append(pair)
    
    return pairedArucoMarkers

def averageRotation(rot1, rot2):
    """
    Average two rotation matrices using OpenCV's Rodrigues conversion.
    """
    # Convert rotation matrices to rotation vectors
    rvec1, _ = cv.Rodrigues(rot1)
    rvec2, _ = cv.Rodrigues(rot2)
    
    # Average the rotation vectors
    rvec_avg = (rvec1 + rvec2) / 2.0
    
    # Convert the averaged rotation vector back to a rotation matrix
    rot_avg, _ = cv.Rodrigues(rvec_avg)
    
    return rot_avg

def locateCenterOfCubes(pair):
    # R_cam2base, t_cam2base = loadRT('R_t_base2cam.npz')
    #load data for the board
    id0 = int(pair['ids'][0])
    id1 = int(pair['ids'][1])
    id0_str = f"{id0:02d}"  # Format as two digits
    id1_str = f"{id1:02d}" 
    root = os.getcwd()
    cvs_dir = os.path.join(root, 'csv')
    csv_filename = os.path.join(cvs_dir, f"positions_plate_{id0_str}-{id1_str}.csv")
    data = []
    with open(csv_filename, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            float_row = [float(value) for value in row]
            data.append(float_row)
        
    T_base2marker0 = pair['T'][0]
    T_base2marker1 = pair['T'][1]

    cubePosSE3 = []

    xyz0 = T_base2marker0[:3, 3]
    xyz1 = T_base2marker1[:3, 3]
    x0, y0, z0 = xyz0[0], xyz0[1], xyz0[2]
    x1, y1, z1 = xyz1[0], xyz1[1], xyz1[2]
    R_base2board = averageRotation(T_base2marker0[:3,:3], T_base2marker1[:3,:3])
    for i in range(1, len(data), 1):
        t_offset =np.array(xyz0) + (R_base2board @ np.array([+0.1, data[i][0]/1000, data[i][1]/1000])).flatten() #0.1 so it is above the playground for now and I do not break anything
        T_base2cube = np.eye(4)
        T_base2cube[:3, :3] = R_base2board
        T_base2cube[:3, 3] = t_offset
        cubePosSE3.append(T_base2cube)
    # print(T_base2marker0)
    # print(cubePosSE3[0])
    return cubePosSE3

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

def drawFoundCubes(img, camera_matrix, dist_coeffs, cubes, T_base2cam):


    for T_base2cube in cubes:
        print(T_base2cube)
        T_cube2base = invert_homogeneous_transform(T_base2cube)
        # print(T_cube2base)
        T_cube2cam = T_cube2base @ T_base2cam
        T_cam2cube = invert_homogeneous_transform(T_cube2cam)
        R_cam2cube = T_cam2cube[:3,:3]
        rvecs, _ = cv.Rodrigues(R_cam2cube)
        tvecs = T_cam2cube[:3,3]
        # break
        cv.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs, tvecs, 0.04)
        


    return img
