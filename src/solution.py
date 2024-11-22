import threading
import numpy as np
from enum import Enum
import time
import cv2 as cv
import os
# from ctu_crs import CRS97
# from basler_camera import BaslerCamera
from utils import loadParams, arucoMarkersFinder

class ArucoType(Enum):
    DICT_4X4_50 = cv.aruco.DICT_4X4_50
    DICT_4X4_100 = cv.aruco.DICT_4X4_100
    DICT_4X4_250 = cv.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv.aruco.DICT_4X4_1000


camMatrix, distCoeff = loadParams('calibration_ciirc.npz')




def locateAllArucoMarkers(img):
    arucoMarkersFinder(img, ArucoType.DICT_4X4_50, camMatrix, distCoeff, 4.0)



if __name__=="__main__":
    root = os.getcwd()
    img_dir = os.path.join(root, 'imgs')
    img_filename = os.path.join(img_dir, f"img{i}_0.png")
    img = cv.imread(img_filename)
    locateAllArucoMarkers(img)