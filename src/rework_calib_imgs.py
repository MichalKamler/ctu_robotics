
import threading
import numpy as np
from enum import Enum
import time
import cv2 as cv
import os
# from ctu_crs import CRS97
# from basler_camera import BaslerCamera
from utils import loadParams, arucoMarkerPoseEstimation

class ArucoType(Enum):
    DICT_4X4_50 = cv.aruco.DICT_4X4_50
    DICT_4X4_100 = cv.aruco.DICT_4X4_100
    DICT_4X4_250 = cv.aruco.DICT_4X4_250
    DICT_4X4_1000 = cv.aruco.DICT_4X4_1000


camMatrix, distCoeff = loadParams('calibration_ciirc.npz')
# print(camMatrix, distCoeff)

exit()

root = os.getcwd()
img_dir = os.path.join(root, 'aruco_calib', 'imgs')
img_data_dir = os.path.join(root, 'aruco_calib', 'img_data')

for i in range(80):
    img_filename = os.path.join(img_dir, f"img{i}_0.png")
    img = cv.imread(img_filename)

    # if img is None:
    #     print("Error: Could not load image.")
    # else:
    #     cv.imshow("Image Window", img)
    #     cv.waitKey(0)  
    #     cv.destroyAllWindows()

    img_aruco, rvec, tvec = arucoMarkerPoseEstimation(img, ArucoType.DICT_4X4_50, camMatrix, distCoeff, 0.06, [0.16, 0.03, 0.0])

    # print(rvec, tvec)

    # if img_aruco is None:
    #     print("Error: Could not load image.")
    # else:
    #     cv.imshow("Image Window", img_aruco)
    #     cv.waitKey(0)  
    #     cv.destroyAllWindows()
    # break

    if rvec is None or tvec is None:
        continue

    img_filename = os.path.join(img_dir, f"img{i}_1.png")
    cv.imwrite(img_filename, img_aruco)

    img_data_filename = os.path.join(img_data_dir, f"data{i}.txt")

    with open(img_data_filename, "w") as file:
        file.write("# rvecs: \n")
        np.savetxt(file, rvec, fmt='%s')

        file.write("# tvecs: \n")
        np.savetxt(file, tvec, fmt='%s')
