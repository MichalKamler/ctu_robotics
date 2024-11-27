import numpy as np

R = np.array([[-9.99996592e-01, -2.59180517e-03, -3.13996969e-04],
 [-2.59178550e-03,  9.99996639e-01, -6.30237828e-05],
 [ 3.14159259e-04, -6.22097552e-05, -9.99999949e-01]])

# R = np.array([[-8.92581856e-04,  1.30794728e-02, -9.99914062e-01],
#               [-5.01241673e-03,  9.99901840e-01,  1.30837873e-02],
#               [ 9.99987039e-01,  5.02366432e-03, -8.26934472e-04]])
sy = -R[2, 0]
sy = np.clip(sy, -1, 1)

# Calculate Euler angles
ry = np.arcsin(sy)  # rotation about y-axis
rx = np.arctan2(R[2, 1], R[2, 2])  # rotation about x-axis
rz = np.arctan2(R[1, 0], R[0, 0])  # rotation about z-axis

# Convert to degrees if needed
rx_deg = np.degrees(rx)
ry_deg = np.degrees(ry)
rz_deg = np.degrees(rz)

# Print results
print(f"r_x (rad): {rx}, r_y (rad): {ry}, r_z (rad): {rz}")
print(f"r_x (deg): {rx_deg}, r_y (deg): {ry_deg}, r_z (deg): {rz_deg}")

Rx = np.array([[1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]])

Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]])

Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]])

R1 = Rz @ Ry @ Rx

print(R)
print()

print(R1)