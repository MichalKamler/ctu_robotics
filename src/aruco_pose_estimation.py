from enum import Enum
import cv2
import numpy as np 
import os 
import matplotlib.pyplot as plt 
from utils import loadParams

class ArucoType(Enum):
    DICT_4X4_50 = cv2.aruco.DICT_4X4_50
    DICT_4X4_100 = cv2.aruco.DICT_4X4_100
    DICT_4X4_250 = cv2.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv2.aruco.DICT_4X4_1000
    DICT_5X5_50 = cv2.aruco.DICT_5X5_50
    DICT_5X5_100 = cv2.aruco.DICT_5X5_100
    DICT_5X5_250 = cv2.aruco.DICT_5X5_250
    DICT_5X5_1000 = cv2.aruco.DICT_5X5_1000
    DICT_6X6_50 = cv2.aruco.DICT_6X6_50
    DICT_6X6_100 = cv2.aruco.DICT_6X6_100
    DICT_6X6_250 = cv2.aruco.DICT_6X6_250
    DICT_6X6_1000 = cv2.aruco.DICT_6X6_1000
    DICT_7X7_50 = cv2.aruco.DICT_7X7_50
    DICT_7X7_100 = cv2.aruco.DICT_7X7_100
    DICT_7X7_250 = cv2.aruco.DICT_7X7_250
    DICT_7X7_1000 = cv2.aruco.DICT_7X7_1000
    DICT_ARUCO_ORIGINAL = cv2.aruco.DICT_ARUCO_ORIGINAL
    DICT_APRILTAG_16h5 = cv2.aruco.DICT_APRILTAG_16h5
    DICT_APRILTAG_25h9 = cv2.aruco.DICT_APRILTAG_25h9
    DICT_APRILTAG_36h10 = cv2.aruco.DICT_APRILTAG_36h10
    DICT_APRILTAG_36h11 = cv2.aruco.DICT_APRILTAG_36h11

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

class ArucoMarkers(): 
    def __init__(self): 
        self.dir = os.path.dirname(os.path.abspath(__file__))

    def arucoMarkerPoseEstimation(self, aruco_type, camera_matrix, dist_coeffs): #for realtime aplication
        print('Detecting ArUco Marker...')

        aruco_side_size = 6.0
        cap = cv2.VideoCapture(0)
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type.value) 
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict,aruco_params)

        world_points = np.array([[0.,0.,0.], # top left
                                 [1.,0.,0.], # top right
                                 [1.,1.,0.], # bottom right
                                 [0.,1.,0.]  # bottom left
        ])
        while True: 
            ret, frame = cap.read() 
        
            if not ret: 
                break 

            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(frame_gray)

            if ids is not None: 
                frame = cv2.aruco.drawDetectedMarkers(frame,corners,ids)

                for corner in corners: 

                    center_x = int(corner[0][:, 0].mean())
                    center_y = int(corner[0][:, 1].mean())

                    success,rvecs,tvecs = cv2.solvePnP(world_points,corner,camera_matrix,dist_coeffs)
                    if not success:
                        continue

                    rotation_matrix, _ = cv2.Rodrigues(rvecs)

                    flip_matrix = np.array([
                                    [0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1]
                                  ], dtype=np.float32)
                    transformed_rotation_matrix = rotation_matrix @ flip_matrix
                    euler_angles = cv2.decomposeProjectionMatrix(np.hstack((transformed_rotation_matrix, tvecs)))[6]
                            
                    tilt_info = f"Center: ({center_x}, {center_y}), Tilt (Roll, Pitch, Yaw): {euler_angles[0][0]:.2f}, {euler_angles[1][0]:.2f}, {euler_angles[2][0]:.2f}"
                    # print(tilt_info)
                    rvecs_transformed, _ = cv2.Rodrigues(transformed_rotation_matrix)

                    R = rotationToMatrix(np.deg2rad(euler_angles[0][0]), np.deg2rad(euler_angles[1][0]), np.deg2rad(euler_angles[2][0]))
                    R_inv = R.T
                    vector_local = R_inv @ tvecs

                    vector_local += np.array([[6.0], [3.0], [0]])
                    vector_new = R @ vector_local


                    # print(tvecs)
                    cv2.putText(frame, tilt_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs_transformed, tvecs, 1) 
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs_transformed, vector_new, 1)

            cv2.imshow('ArUco Detection',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break



def runArucoMarkerPoseEstimation(aruco_type): 
    aruco_marker = ArucoMarkers() 
    # Logitech Camera Calibration (Need to calibrate for your own camera)
    camMatrix, distCoeff = loadParams()
    aruco_marker.arucoMarkerPoseEstimation(aruco_type, camMatrix, distCoeff)

if __name__ == '__main__': 
    # run_generate_aruco_marker(ArucoType.DICT_6X6_250,marker_id=0,marker_width_pixels=200)
    runArucoMarkerPoseEstimation(ArucoType.DICT_4X4_50) 