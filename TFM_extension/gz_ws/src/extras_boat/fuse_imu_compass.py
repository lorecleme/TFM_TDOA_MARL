# Markus Buchholz

from pymavlink import mavutil
import time
import numpy as np

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.estimate_error = 1.0

    def update(self, measurement):
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.estimate_error = (1 - kalman_gain) * self.estimate_error + abs(self.estimate) * self.process_variance

        return self.estimate

def get_imu_data(master):
    data = {}
    while True:
        message = master.recv_match(type=['ATTITUDE', 'RAW_IMU'], blocking=True)
        if message:
            msg_type = message.get_type()
            data[msg_type] = message.to_dict()
            
            if "ATTITUDE" in data and "RAW_IMU" in data:
                return data

def read_compass_heading(master):
    while True:
        message = master.recv_match(type='VFR_HUD', blocking=True)
        if message:
            heading = message.heading % 360
            return heading

def main():
  
    master = mavutil.mavlink_connection('udpin:192.168.2.1:14550')
   
    master.wait_heartbeat()

    roll_kf = KalmanFilter(process_variance=1e-5, measurement_variance=1e-2)
    heading_kf = KalmanFilter(process_variance=1e-5, measurement_variance=1e-2)

    while True:
        data = get_imu_data(master)

        if "ATTITUDE" not in data:
            print("No ATTITUDE data")
            continue

        if "RAW_IMU" not in data:
            print("No RAW_IMU data")
            continue

        imu_data = data["RAW_IMU"]
        attitude_data = data["ATTITUDE"]

        roll = attitude_data["roll"]
        compass_heading = read_compass_heading(master)

        fused_roll = roll_kf.update(roll)
        fused_heading = heading_kf.update(compass_heading)

        print(f"Fused Roll: {np.degrees(fused_roll):.2f} degrees")
        print(f"Fused Heading: {fused_heading:.2f} degrees")
        time.sleep(1)

if __name__ == "__main__":
    main()
