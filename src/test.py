import threading
from enum import Enum
import numpy as np
import cv2 as cv
import os
from ctu_crs import CRS97
from basler_camera import BaslerCamera
from utils import loadParams, arucoMarkerPoseEstimation

class ArucoType(Enum):
    DICT_4X4_50 = cv.aruco.DICT_4X4_50
    DICT_4X4_100 = cv.aruco.DICT_4X4_100
    DICT_4X4_250 = cv.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv.aruco.DICT_4X4_1000
    DICT_5X5_50 = cv.aruco.DICT_5X5_50
    DICT_5X5_100 = cv.aruco.DICT_5X5_100
    DICT_5X5_250 = cv.aruco.DICT_5X5_250
    DICT_5X5_1000 = cv.aruco.DICT_5X5_1000
    DICT_6X6_50 = cv.aruco.DICT_6X6_50
    DICT_6X6_100 = cv.aruco.DICT_6X6_100
    DICT_6X6_250 = cv.aruco.DICT_6X6_250
    DICT_6X6_1000 = cv.aruco.DICT_6X6_1000
    DICT_7X7_50 = cv.aruco.DICT_7X7_50
    DICT_7X7_100 = cv.aruco.DICT_7X7_100
    DICT_7X7_250 = cv.aruco.DICT_7X7_250
    DICT_7X7_1000 = cv.aruco.DICT_7X7_1000
    DICT_ARUCO_ORIGINAL = cv.aruco.DICT_ARUCO_ORIGINAL
    DICT_APRILTAG_16h5 = cv.aruco.DICT_APRILTAG_16h5
    DICT_APRILTAG_25h9 = cv.aruco.DICT_APRILTAG_25h9
    DICT_APRILTAG_36h10 = cv.aruco.DICT_APRILTAG_36h10
    DICT_APRILTAG_36h11 = cv.aruco.DICT_APRILTAG_36h11


img = cv.imread('aruco_calib/imgs/image.png')

camMatrix, distCoeff = loadParams('calibration_ciirc.npz')

img_aruco, rvec, tvec = arucoMarkerPoseEstimation(img, ArucoType.DICT_4X4_50, camMatrix, distCoeff, 8.0, [16.0, 4.0, 0])

print(rvec)
print(tvec)

# Check if the image was loaded successfully
if img is None:
    print("Failed to load image. Check the file path.")
else:
    print("Image loaded successfully!")

# Display the image
cv.imshow('Loaded Image', img)
cv.waitKey(0)  # Wait for a key press to close the window
cv.destroyAllWindows()
