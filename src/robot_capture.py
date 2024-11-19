import threading
import numpy as np
import os
from ctu_crs import CRS97

monitor_terminal_cmds = True


def rotationMatrixToEulerAngles(R):
    """
    Convert a rotation matrix R to Euler angles (roll, pitch, yaw).
    The rotation matrix is assumed to be a 3x3 matrix.
    """
    # Pitch (theta) around Y-axis
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    # Roll (phi) around X-axis
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    # Yaw (psi) around Z-axis
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # Convert radians to degrees if needed
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg

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

def filterSolOutBounds(robot, q_sol):
    valid_q_sol = []
    q_min = robot.q_min
    q_max = robot.q_max
    for i in range(len(q_sol)):
        valid = True
        for j in range(len(q_sol[i])):
            if q_sol[i][j]<q_min[j] or q_sol[i][j]>q_max[j]:
                valid = False
        if valid:
            valid_q_sol.append(q_sol[i])
    return valid_q_sol

def help(robot):
    print("reset - resets motors of the robot")
    print("cage - waits for the robot to stop moving and releases the robot so you can safely enter the cage")
    print("end - moves robot to home and terminates the connection")
    # "moveGpripperXYZ": moveGpripperXYZ,
    # "setGripperDown": setGripperDown,
    # "printPose": printPose,
    # "moveBase" : moveBase, 

def printPose(robot):
    q = robot.get_q()
    pose = robot.fk(q)
    print(pose)
    print(q)
    print(robot.q_min)
    print(robot.q_max)
    # roll, pitch, yaw = rotationMatrixToEulerAngles(pose[:3, :3])
    # print("roll, pitch, yaw", roll, pitch, yaw)
    # print(rotationToMatrix(roll,pitch,yaw))

def setGripperDown(robot):
    print("setGripperDown is running")
    q0 = robot.get_q()

    robot.move_to_q(q0 + [0.0, 0.0, 0.0, 0.0, -np.pi/2, 0.0])
    robot.wait_for_motion_stop()
    print("setGripperDown completed")

def start(robot):
    moveGripperXYZ(robot, 0, 0, -0.2)
    setGripperDown(robot)

def moveGripperXYZ(robot, x, y, z):
    print("moveGripperXYZ is running")
    q0 = robot.get_q()
    pose = robot.fk(q0)

    pose[:3, 3] += np.array([x, y, z])

    possible_q_config = robot.ik(pose)
    valid_q_sol = filterSolOutBounds(robot, possible_q_config)
    if len(valid_q_sol)>0:
        closest_solution = min(valid_q_sol, key=lambda q: np.linalg.norm(q - q0))
        print(closest_solution)
        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()
    else:
        print("No solution found for the given x y z")
    print("moveGripperXYZ completed")

def reset(robot):
    print("reset is running!")
    robot.reset_motors()
    print("reset completed")

def cage(robot):
    print("cage is running!")
    if robot.in_motion():
        robot.wait_for_motion_stop()
    robot.release()
    print("robot released - OK to go in cage")
    print("cage completed")

def moveBase(robot, deg):
    print("moveBase is running")
    rad = np.deg2rad(deg)
    q0 = robot.get_q()
    robot.move_to_q(q0 + [rad, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.wait_for_motion_stop()
    print("moveBase completed")

def rotateGripper(robot, deg):
    print("rotateGripper is running")
    rad = np.deg2rad(deg)
    q0 = robot.get_q()
    robot.move_to_q(q0 + [0.0, 0.0, 0.0, 0.0, 0.0, rad])
    robot.wait_for_motion_stop()
    print("rotateGripper completed")

def captureImgAndPose()


def saveQ(robot, i):
    print("saveQ is running")
    directory = "robot_poses"
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"output{i}.txt")


    q = robot.get_q() # ndarray
    pose = robot.fk(q) # 4x4 array
    xyz = robot.fk_flange_pos(q) #ndarray

    with open(filename, "w") as file: 
        file.write("# q: \n")
        np.savetxt(file, [q], fmt='%s')

        file.write("\n# Pose (4x4 matrix): \n")
        np.savetxt(file, pose, fmt='%.6f')

        file.write("\n# XYZ: \n")
        np.savetxt(file, [xyz], fmt='%s')
    print("saveQ completed")

def end(robot):
    print("end is running!")
    robot.soft_home()
    robot.close()
    monitor_terminal_cmds = False
    print("end completed!")

functions = {
    "help": help,
    "start": start,
    "moveGripperXYZ": moveGripperXYZ,
    "setGripperDown": setGripperDown,
    "printPose": printPose,
    "moveBase": moveBase, 
    "rotateGripper": rotateGripper,
    "saveQ": saveQ,
    "reset": reset,
    "cage": cage,
    "end": end,
}

def parse_and_execute(robot, command):
    parts = command.split()
    
    func_name = parts[0]
    if func_name in functions:
        # Prepare arguments based on the function signature
        try: 
            args = []
            for arg in parts[1:]:
                try:
                    # Extract arguments as needed
                    args.append(float(arg))
                except Exception as e:
                    print(f"Error: {e}")
            functions[func_name](robot, *args)  # Call the function with unpacked arguments
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Unknown function: {func_name}")

def monitor_terminal(robot):
    print("Type the function name (e.g., func1, func2) and press Enter to execute.")
    while monitor_terminal_cmds:
        user_input = input("Enter function name: ").strip()
        parse_and_execute(robot, user_input)


# robot.fk_flange_pos(q: ArrayLike) -> x y z of the gripper with regards to the base 0 0 0

# robot.get_q(self) -> np.ndarray -> Get current joint configuration.

# robot.move_to_q(q: ArrayLike)

# robot.fk(self, q: ArrayLike) -> np.ndarray: -> pose represented as 4x4 matrix SE3 wrt base 

# robot.ik(self, pose: np.ndarray) -> list[np.ndarray]:


if __name__ == "__main__":
    robot = CRS97()
    robot.initialize() # performs soft home!!
    monitor_terminal(robot)
