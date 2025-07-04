import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyproj import Proj, transform

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_multiply(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    a = 6378137.0  # WGS-84 ellipsoid params
    f = 1 / 298.257223563
    e_sq = f * (2 - f)

    def to_ecef(lat, lon, alt):
        N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = ((1 - e_sq) * N + alt) * np.sin(lat)
        return np.array([x, y, z])

    def ecef_to_enu(x, y, z, lat0, lon0, alt0):
        x0, y0, z0 = to_ecef(lat0, lon0, alt0)
        dx = x - x0
        dy = y - y0
        dz = z - z0
        slat = np.sin(lat0)
        clat = np.cos(lat0)
        slon = np.sin(lon0)
        clon = np.cos(lon0)
        t = np.array([[-slon, clon, 0], [-slat*clon, -slat*slon, clat], [clat*clon, clat*slon, slat]])
        return t @ np.array([dx, dy, dz])

    ecef = to_ecef(lat, lon, alt)
    return ecef_to_enu(*ecef, lat0, lon0, alt0)

class EKF_IMU_Bias:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(16)  # [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bgx, bgy, bgz, bax, bay, baz]
        acc0 = acc[0] / np.linalg.norm(acc[0])
        pitch = np.arcsin(acc0[0])
        roll = np.arctan2(-acc0[1], -acc0[2])
        yaw = 0.0
        q0 = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        self.x[6:10] = q0  # initialize orientation from accelerometer
        self.P = np.eye(15) * 0.1
        self.Q = np.diag([
            1e-4]*3 + [1e-4]*3 + [1e-8]*3 + [1e-4]*3 + [1e-2]*3
        )
        self.Rp = np.eye(3) * 0.25  # position measurement noise
        self.Rv = np.eye(3) * 0.01  # velocity measurement noise
        self.I = np.eye(15)

    def predict(self, acc_m, gyro_m):
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        bg = self.x[10:13]
        ba = self.x[13:16]

        g = np.array([0, 0, -9.81])
        q = normalize_quaternion(q)
        Rwb = R.from_quat(q).as_matrix()
        a_world = Rwb @ (acc_m - ba)
        self.x[0:3] += v * self.dt
        self.x[3:6] += (a_world + g) * self.dt

        omega = gyro_m - bg
        omega_quat = np.concatenate([[0], omega])
        dq = 0.5 * quaternion_multiply(q, omega_quat)
        self.x[6:10] += dq * self.dt
        self.x[6:10] = normalize_quaternion(self.x[6:10])

        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 12:15] = -Rwb * self.dt
        self.P = F @ self.P @ F.T + self.Q

    def update_position(self, pos_meas):
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        y = pos_meas - self.x[0:3]
        S = H @ self.P @ H.T + self.Rp
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.x[0:3] += dx[0:3]
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]
        self.P = (self.I - K @ H) @ self.P

    def update_velocity(self, vel_meas):
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        y = vel_meas - self.x[3:6]
        S = H @ self.P @ H.T + self.Rv
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.x[3:6] += dx[3:6]
        self.x[10:13] += dx[9:12]
        self.x[13:16] += dx[12:15]

        # Orientation update (small angle correction)
        delta_theta = dx[6:9]  # assume small-angle rotation correction in error state
        dq = np.concatenate([[1.0], 0.5 * delta_theta])
        q = self.x[6:10]
        self.x[6:10] = normalize_quaternion(quaternion_multiply(q, dq))

        self.P = (self.I - K @ H) @ self.P

    def get_state(self):
        return self.x.copy()

imu_df = pd.read_csv('/Users/subhash/Documents/projects/AIRL/IMU_bias_estimation/A-KIT/Data/Trajectory1/IMU_trajectory1.csv')
vel_df = pd.read_csv('/Users/subhash/Documents/projects/AIRL/IMU_bias_estimation/A-KIT/Data/Trajectory1/GT_trajectory1.csv')

lat0, lon0, alt0 = vel_df.iloc[0]["Latitude [rad]"], vel_df.iloc[0]["Longitude [rad]"], vel_df.iloc[0]["Altitude [m]"]

enu_pos = np.array([geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0) for lat, lon, alt in zip(
    vel_df["Latitude [rad]"], vel_df["Longitude [rad]"], vel_df["Altitude [m]"]
)])

vel_interp = pd.DataFrame()
vel_interp['Time'] = imu_df['Time [s]']
for col in ['V North [m/s]', 'V East [m/s]', 'V Down [m/s]']:
    vel_interp[col] = np.interp(imu_df['Time [s]'], vel_df['Time [s]'], vel_df[col])
for i, label in enumerate(['X', 'Y', 'Z']):
    vel_interp[f'ENU {label} [m]'] = np.interp(imu_df['Time [s]'], vel_df['Time [s]'], enu_pos[:, i])

acc = imu_df[['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]']].values
gyro = imu_df[['GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']].values
vel = vel_interp[['V North [m/s]', 'V East [m/s]', 'V Down [m/s]']].values
pos = vel_interp[['ENU X [m]', 'ENU Y [m]', 'ENU Z [m]']].values
times = imu_df['Time [s]'].values

ekf = EKF_IMU_Bias(dt=0.01)
pos_est, vel_est, bias_acc_est, bias_gyro_est = [], [], [], []

for i in range(len(times)):
    if i % 500 == 0:
        state = ekf.get_state()
        print(f"t={times[i]:.2f}s | q = {state[6:10]} | bg = {state[10:13]}")
    ekf.predict(acc[i], gyro[i])
    if i % 100 == 0:
        ekf.update_velocity(vel[i])
        ekf.update_position(pos[i])
    state = ekf.get_state()
    pos_est.append(state[0:3])
    vel_est.append(state[3:6])
    bias_gyro_est.append(state[10:13])
    bias_acc_est.append(state[13:16])

pos_est = np.array(pos_est)
vel_est = np.array(vel_est)
bias_gyro_est = np.array(bias_gyro_est)
bias_acc_est = np.array(bias_acc_est)

plt.figure(figsize=(12, 4))
plt.title("Estimated Position vs Ground Truth (ENU)")
for i, axis in enumerate(['X', 'Y', 'Z']):
    plt.subplot(1, 3, i+1)
    plt.plot(times, pos[:, i], label='Measured')
    plt.plot(times, pos_est[:, i], label='Estimated', linestyle='--')
    plt.ylabel(f'Position {axis} [m]')
    plt.xlabel('Time [s]')
    plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.suptitle("Estimated Accelerometer Bias [m/s²]")
for i, axis in enumerate(['X', 'Y', 'Z']):
    plt.subplot(1, 3, i+1)
    plt.plot(times, bias_acc_est[:, i])
    plt.xlabel('Time [s]')
    plt.ylabel(f'Bias {axis}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.suptitle("Estimated Gyroscope Bias [rad/s]")
for i, axis in enumerate(['X', 'Y', 'Z']):
    plt.subplot(1, 3, i+1)
    plt.plot(times, bias_gyro_est[:, i])
    plt.xlabel('Time [s]')
    plt.ylabel(f'Bias {axis}')
plt.tight_layout()
plt.show()
