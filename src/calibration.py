import threading
from ctu_crs import CRS97

def func1():
    print("func1 is running!")

def func2():
    print("func2 is running!")

def func3():
    print("func3 is running!")

functions = {
    "func1": func1,
    "func2": func2,
    "func3": func3,
}

def monitor_terminal():
    print("Type the function name (e.g., func1, func2) and press Enter to execute.")
    while True:
        user_input = input("Enter function name: ").strip()
        if user_input in functions:
            functions[user_input]()
        else:
            print(f"Unknown command: {user_input}")

# if __name__ == "__main__":
    # robot = CRS97()  # set argument tty_dev=None if you are not connected to robot,
    # robot.initialize()  # initialize connection to the robot, perform hard and soft home
    # q = robot.get_q()  # get current joint configuration
    # robot.move_to_q(q + [0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # move robot all values in radians
    # robot.wait_for_motion_stop() # wait until the robot stops
    # robot.close()  # close the connection

if __name__ == "__main__":
    monitor_terminal()
