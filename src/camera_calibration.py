import cv2 as cv
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from basler_camera import BaslerCamera

distortionCameraMatrix = np.array([[800, 0, 320],
                         [0, 800, 240],
                         [0, 0, 1]])

# Distortion coefficients for barrel or pincushion distortion
# k1, k2 are the radial distortion coefficients
# For example, k1 = -0.5 for pincushion or k1 = 0.5 for barrel distortion
distortionDistCoeffs = np.array([0.5, 0.0, 0.0, 0.0, 0.0])


def getWidhtHeight(cap):
    got_w_h = False
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image. No width and height")
    else: 
        height, width = frame.shape[:2]
    return width, height

def delete_all_images():
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages', 'calibration')
    image_files = glob.glob(os.path.join(calibrationDir, '*.jpg'))  # Change '*.jpg' to '*.png' or other patterns as needed

    if not image_files:
        print(f"No images found in {calibrationDir}.")
        return

    for img_file in image_files:
        try:
            os.remove(img_file)
            print(f"Deleted {img_file}")
        except Exception as e:
            print(f"Error deleting {img_file}: {e}")

def displayImg(cap, width, height, camMatrix =None, distCoeff =None):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages', 'calibration')

    if not os.path.exists(calibrationDir):
        os.makedirs(calibrationDir)
        print(f"Created directory: {calibrationDir}")

    counter = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        distorted_image = cv.undistort(frame, distortionCameraMatrix, distortionDistCoeffs)
                
        # Display the resulting frame
        if camMatrix is None and distCoeff is None:
            gray = cv.cvtColor(distorted_image, cv.COLOR_BGR2GRAY)
            # Initialize the ArUco dictionary and detector
            aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
            parameters = cv.aruco.DetectorParameters_create()
            # Detect markers in the image
            corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            # Draw the markers on the image
            if len(corners) > 0:
                cv.aruco.drawDetectedMarkers(distorted_image, corners, ids)

            cv.imshow('Camera Feed Distorted', distorted_image)
        else: 
            undistorted_image = removeDistortion(camMatrix=camMatrix, distCoeff=distCoeff, img=distorted_image, w=width, h=height)

            gray = cv.cvtColor(undistorted_image, cv.COLOR_BGR2GRAY)
            # Initialize the ArUco dictionary and detector
            aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)
            parameters = cv.aruco.DetectorParameters_create()
            # Detect markers in the image
            corners, ids, rejected_img_points = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            # Draw the markers on the image
            if len(corners) > 0:
                cv.aruco.drawDetectedMarkers(undistorted_image, corners, ids)

            cv.imshow('Camera Feed Undistorted', undistorted_image) 


        # Break the loop if the user presses the 'q' key
        # Check if the user presses the 'q' key to quit
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Check if the spacebar (ASCII value 32) is pressed to save the image
        elif key == 32:
            # Save the distorted image to the specified folder
            filename = os.path.join(calibrationDir, f"img{counter}.jpg")
            cv.imwrite(filename, distorted_image)
            print(f"Image saved as {filename}")
            counter += 1

    # Release the capture object and close any open windows
    cv.destroyAllWindows()

def displayFeed(cam, camMatrix =None, distCoeff =None):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages', 'calibration')
    if not os.path.exists(calibrationDir):
        os.makedirs(calibrationDir)
        print(f"Created directory: {calibrationDir}")

    while True:
        img = cam.grab_image()
        height, width = img.shape[:2]
        if (img is not None) and (img.size > 0):
            # Display the resulting frame
            if camMatrix is None and distCoeff is None:
                cv.imshow('Camera Feed NOT CALIBRATED', img)
            else: 
                calibrated_img = removeDistortion(camMatrix=camMatrix, distCoeff=distCoeff, img=img, w=width, h=height)
                cv.imshow('Camera Feed Calibrated', calibrated_img) 

            # Break the loop if the user presses the 'q' key
            # Check if the user presses the 'q' key to quit
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Check if the spacebar (ASCII value 32) is pressed to save the image
            elif key == 32:
                # Save the distorted image to the specified folder
                filename = os.path.join(calibrationDir, f"img{counter}.jpg")
                cv.imwrite(filename, img)
                print(f"Image saved as {filename}")
                counter += 1
        else:
            print("The image was not captured.")
                
    # Release the capture object and close any open windows
    cv.destroyAllWindows()

def removeDistortion(camMatrix, distCoeff, img, w, h):
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix, distCoeff, (w,h), 1, (w,h))
    imgUndist = cv.undistort(img, camMatrix, distCoeff, None, camMatrixNew)
    return imgUndist

def calibrate(showPics=True):
    # Read Image
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'demoImages//calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    # Initialize 
    nRows = 8
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = []

    # Find corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)
        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11,11),(-1,-1),termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(1000)
    cv.destroyAllWindows()

    # Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print('Camera Matrix: \n', camMatrix)
    print('Reproj Error (pixels): {:.4f}'.format(repError))

    # Save Calibration Parameters
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder, 'calibration.npz')
    np.savez(paramPath, repError=repError, camMatrix=camMatrix, distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

    return camMatrix, distCoeff

def loadParams():
    curFolder = os.path.dirname(os.path.abspath(__file__))
    # Load Calibration Parameters
    paramPath = os.path.join(curFolder, 'calibration.npz')

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

if __name__ == "__main__":
    ciirc = False
    capture_and_calibrate_distorted_img = False

    if ciirc:
        camera: BaslerCamera = BaslerCamera()
        camera.connect_by_name("camera-crs97")
        camera.set_parameters()
        camera.start()
        displayFeed(camera, )

    else:
        cap = cv.VideoCapture(0)
        width, height = getWidhtHeight(cap)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()  

        if capture_and_calibrate_distorted_img:
            delete_all_images()
            displayImg(cap, width, height)
            camMatrix, distCoeff = calibrate(True)
            displayImg(cap, width, height, camMatrix, distCoeff)
        else:
            camMatrix, distCoeff = loadParams()
            displayImg(cap, width, height, camMatrix, distCoeff)
        cap.release()


