import numpy as np

# R = np.array([[-9.99996592e-01, -2.59180517e-03, -3.13996969e-04],
#  [-2.59178550e-03,  9.99996639e-01, -6.30237828e-05],
#  [ 3.14159259e-04, -6.22097552e-05, -9.99999949e-01]])

# R = np.array([[ 6.28298834e-05, -1.00534977e-04,  9.99999993e-01], #pose at home
#  [ 2.20000957e-05,  9.99999995e-01,  1.00533595e-04],
#  [-9.99999998e-01,  2.19937790e-05,  6.28320949e-05]])

# R = np.array([[-0.03060615,  0.11029646, -0.99342738],
#  [ 0.05698358,  0.99246906,  0.10843448],
#  [ 0.99790587, -0.05329029, -0.03666074]])

# R = np.array([[-9.99999994e-01, -1.57079633e-05,  1.09955743e-04  ], # home
#  [-1.57079632e-05,  1.00000000e+00,  1.72718089e-09],
#  [-1.09955743e-04,  1.22437747e-16, -9.99999994e-01]])

R = np.array([[5.37453672e-09,  9.99999990e-01, 1.41371669e-04],
 [ 1.00000000e+00, -9.77188248e-10, -3.11048777e-05],
 [-3.11048773e-05,  1.41371669e-04, -9.99999990e-01]])

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