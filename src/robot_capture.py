import threading
import numpy as np
from ctu_crs import CRS97

monitor_terminal_cmds = True


def rotation_matrix_to_euler_angles(R):
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

def help(robot):
    print("reset - resets motors of the robot")
    print("cage - waits for the robot to stop moving and releases the robot so you can safely enter the cage")
    print("end - moves robot to home and terminates the connection")

def printPose(robot):
    q = robot.get_q()
    pose = robot.fk(q)
    print(pose)

def setGripperDown(robot):
    print("setGripperDown is running")
    q0 = robot.get_q()
    pose = robot.fk(q0)
    # R = pose[:3, :3]  # Rotation matrix (3x3)
    t = pose[:3, 3]   # Translation vector (3x1)
    R = rotationToMatrix(0, 0, np.pi)
    pose[:3, :3] = R
    possible_q_config = robot.ik(pose)
    if len(possible_q_config)>0:
        closest_solution = min(possible_q_config, key=lambda q: np.linalg.norm(q - q0))
        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()
    else:
        print("No possible solution for this gripper x y z position")
    print("setGripperDown completed")



def moveGpripperXYZ(robot, x, y, z):
    print("moveGpripperXYZ is running")
    q0 = robot.get_q()
    pose = robot.fk(q0)

    R = pose[:3, :3]  # Rotation matrix (3x3)
    t = pose[:3, 3]   # Translation vector (3x1)

    R_inv = R.T
    vec_local = R_inv @ t
    vec_local += np.array([[x], [y], [z]])
    vec_new = R @ vec_local

    desired_pose = np.copy(pose) 
    desired_pose[:3, 3] = vec_new 

    possible_q_config = robot.ik(desired_pose)
    if len(possible_q_config)>0:
        closest_solution = min(possible_q_config, key=lambda q: np.linalg.norm(q - q0))
        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()
    else:
        print("No solution found for the given x y z")
    print("moveGpripperXYZ completed")

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


def end(robot):
    print("end is running!")
    robot.soft_home()
    robot.close()
    monitor_terminal_cmds = False
    print("end completed!")

functions = {
    "help": help,
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
            # Extract arguments as needed
            args = [float(arg) if arg.isdigit() else arg for arg in parts[1:]]
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
    robot.initialize()
    monitor_terminal(robot)
