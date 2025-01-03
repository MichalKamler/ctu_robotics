import numpy as np
from enum import Enum
import time
import cv2 as cv
import os
from utils2 import loadCamDist, arucoMarkersFinder, pairUpAruco, locateCenterOfCubes, drawFoundCubes, loadRT

camMatrix, distCoeff = loadCamDist('npz/calibration_ciirc.npz')
R_base2cam, t_base2cam = loadRT('npz/R_t_base2cam.npz')

# src = os.getcwd() 
# paramPath = os.path.join(src, 'npz/R_t_base2cam.npz')
# t_base2cam[1] += -0.009 
# np.savez(paramPath, R=R_base2cam, t=t_base2cam)


T_base2cam = np.eye(4)
T_base2cam[:3, :3] = R_base2cam
T_base2cam[:3, 3] = t_base2cam.flatten()

monitor_terminal_cmds = True


def locateAllCubes(img):

    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (w,h), 1, (w,h))
    print(newcameramtx, roi)
    # undistort
    dst = cv.undistort(img, camMatrix, distCoeff, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    img, allT_base2marker, ids = arucoMarkersFinder(img, camMatrix, distCoeff, 0.036)


    cubesList = []
    if len(ids)>0:
        pairs = pairUpAruco(allT_base2marker, ids)
        for pair in pairs:
            cubes = locateCenterOfCubes(pair)
            img = drawFoundCubes(img, camMatrix, distCoeff, cubes, T_base2cam)
            cubesList.append(cubes)


    if img is None:
        print("Error: Could not load image.")
    else:
        cv.namedWindow("Image Window", cv.WINDOW_NORMAL)
        cv.resizeWindow("Image Window", 1200, 800)
        cv.imshow("Image Window", img)
        cv.waitKey(0)  
        cv.destroyAllWindows()

   
    # # cv.imwrite('calibresult.png', dst)
    # if dst is None:
    #     print("Error: Could not load image.")
    # else:
    #     cv.namedWindow("Image Window", cv.WINDOW_NORMAL)
    #     cv.resizeWindow("Image Window", 1200, 800)
    #     cv.imshow("Image Window", dst)
    #     cv.waitKey(0)  
    #     cv.destroyAllWindows()

    print(cubesList)

    
    return cubesList, pairs


if __name__=="__main__":
    np.set_printoptions(precision=10, suppress=True)
    print(camMatrix, distCoeff)
    print(T_base2cam)
    data = np.load('T_base_to_camera_optimized.npy')

    # Print the contents of the file
    print(data)
    image = cv.imread('test_at_home.jpg')
    locateAllCubes(image)
    