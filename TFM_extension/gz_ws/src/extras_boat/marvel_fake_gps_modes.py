from pymavlink import mavutil
import time
import threading

class BoatController:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.conn = None
        self.running = False

    def connect(self):
        # Create the connection
        if not self.conn:
            self.conn = mavutil.mavlink_connection(self.connection_string)
            # Wait for a heartbeat before sending commands
            self.conn.wait_heartbeat()
            print("Connected to vehicle.")

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Disconnected from vehicle.")

    def arm_vehicle(self):
        # Attempt to arm the vehicle only if position estimates are being sent
        self.connect()
        print("Attempting to arm the vehicle...")

        # Disable pre-arm checks
        self.set_param('ARMING_CHECK', 0)

        # Wait and check for arming confirmation
        timeout = time.time() + 10  # 10 second timeout for arming
        armed = False

        while not armed and time.time() < timeout:
            self.conn.mav.command_long_send(
                self.conn.target_system,
                self.conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1, 0, 0, 0, 0, 0, 0
            )
            message = self.conn.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if message:
                armed = message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                if armed:
                    print('Vehicle is armed!')
                    break
            time.sleep(1)

        if not armed:
            print('Failed to arm the vehicle. Check pre-arm conditions and messages.')

    def disarm_vehicle(self):
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # Confirmation
            0,  # Param1: 0 to disarm
            0, 0, 0, 0, 0, 0)

        # Check if the vehicle has disarmed
        print("Disarming the vehicle...")
        timeout = time.time() + 10  # 10 seconds timeout
        disarmed = False

        while not disarmed and time.time() < timeout:
            heartbeat = self.conn.recv_match(type='HEARTBEAT', blocking=True)
            if heartbeat:
                disarmed = not (heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if disarmed:
                    print('Vehicle is disarmed!')
                    break

        if not disarmed:
            print('Failed to disarm the vehicle. Check vehicle status and conditions.')
        

    def set_mode(self, mode):
        self.connect()
        """Set the vehicle mode"""
        mode_id = self.conn.mode_mapping()[mode]

        # Print current status before setting mode
        self.print_status(f"before setting to {mode}")

        # Set the specified mode
        self.conn.mav.set_mode_send(
            self.conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)

        # Wait for mode change confirmation
        timeout = time.time() + 10  # 10 second timeout for mode change
        while time.time() < timeout:
            message = self.conn.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if message and message.custom_mode == mode_id:
                print(f"Mode successfully changed to {mode}")
                break
            time.sleep(1)
        else:
            print(f"Failed to change mode to {mode}")

    def set_guided_mode(self):
        """Set guided mode"""
        self.set_mode('GUIDED')

    def set_loiter_mode(self):
        """Set loiter mode"""
        self.set_mode('LOITER')

    def set_auto_mode(self):
        """Set auto mode"""
        self.set_mode('AUTO')

    def set_manual_mode(self):
        """Set manual mode"""
        self.set_mode('MANUAL')

    def get_current_mode(self):
        """Get the current mode of the vehicle"""
        self.connect()
        message = self.conn.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        current_mode = None
        if message:
            custom_mode = message.custom_mode
            for mode, mode_id in self.conn.mode_mapping().items():
                if mode_id == custom_mode:
                    current_mode = mode
                    break
        return current_mode

    def print_status(self, description):
        print(f"Current status {description}:")
        message = self.conn.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        if message:
            print(message.to_dict())
        else:
            print("No heartbeat message received.")

    def set_param(self, param_id, param_value):
        """
        Set a parameter value on the vehicle.
        """
        self.conn.mav.param_set_send(
            self.conn.target_system,
            self.conn.target_component,
            param_id.encode('utf-8'),
            param_value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        print(f"Set {param_id} to {param_value}")

    def send_vision_position_estimate(self, x, y, z, roll, pitch, yaw):
        """
        Send vision position estimate data to the vehicle.
        """
        current_time_us = int(time.time() * 1e6)
        tracker_confidence = 3  # Simulate a tracking confidence level (1-3)
        cov_scale = pow(10, 3 - tracker_confidence)
        covariance = [0.01 * cov_scale] * 21  # Simplified diagonal covariance

        self.conn.mav.vision_position_estimate_send(
            current_time_us,  # Timestamp (microseconds since UNIX epoch)
            x,                # Global X position
            y,                # Global Y position
            z,                # Global Z position
            roll,             # Roll angle in radians
            pitch,            # Pitch angle in radians
            yaw,              # Yaw angle in radians
            covariance        # Covariance matrix upper right triangular (first six rows of 6x6 matrix)
        )
        # Removed print statement to avoid cluttering the menu

    def set_ekf_home(self, latitude, longitude, altitude):
        """
        Set the EKF home position.
        """
        q = [1, 0, 0, 0]  # Unit quaternion for no rotation

        self.conn.mav.set_gps_global_origin_send(
            self.conn.target_system,
            int(latitude * 1e7),
            int(longitude * 1e7),
            int(altitude * 1000)
        )
        self.conn.mav.set_home_position_send(
            self.conn.target_system,
            int(latitude * 1e7),
            int(longitude * 1e7),
            int(altitude * 1000),
            0, 0, 0,  # x, y, z positions (local frame)
            q,        # q (w, x, y, z quaternion components)
            0, 0, 0,  # approach x, y, z
        )
        print(f"Set EKF home position: lat={latitude}, lon={longitude}, alt={altitude}")

    def start_vision_position_estimate(self, x, y, z, roll, pitch, yaw):
        """Start sending vision position estimates in a separate thread."""
        self.running = True
        def send_estimates():
            while self.running:
                self.send_vision_position_estimate(x, y, z, roll, pitch, yaw)
                time.sleep(1)
        
        self.vision_thread = threading.Thread(target=send_estimates)
        self.vision_thread.start()

    def stop_vision_position_estimate(self):
        """Stop sending vision position estimates."""
        self.running = False
        if self.vision_thread.is_alive():
            self.vision_thread.join()

    def _get_current_position(self):
        """Get the current vision position estimate (simulated)."""
        # This method would typically retrieve the current position from the vehicle
        # For this simulation, we're just returning the fixed fake position
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def get_current_position(self):
        print("Waiting for GPS position...")
        while True:
            msg = self.conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            if msg:
                initial_lat = msg.lat / 1e7
                initial_lon = msg.lon / 1e7
                print(f"Initial position: lat={initial_lat}, lon={initial_lon}")
                return initial_lat, initial_lon

def menu():
    print("\nBoat Controller Menu")
    print("1. Check current mode")
    print("2. Set mode to MANUAL")
    print("3. Set mode to GUIDED")
    print("4. Set mode to AUTO")
    print("5. Set mode to LOITER")
    print("6. Arm vehicle")
    print("7. Disarm vehicle")
    print("8. Get current position")
    print("0. Exit")

if __name__ == "__main__":
    boat_controller = BoatController('udpin:0.0.0.0:14550')
    #boat_controller = BoatController('udpin:192.168.2.1:14550')

    # Connect and set initial parameters
    boat_controller.connect()
    params = {
        'AHRS_EKF_TYPE': 2,
        'EK2_ENABLE': 1,
        'EK3_ENABLE': 0,
        'GPS_TYPE': 0,
        'EK2_GPS_TYPE': 3,
        'EK2_POSNE_M_NSE': 0.1,
        'EK2_VELD_M_NSE': 0.1,
        'EK2_VELNE_M_NSE': 0.1,
        'COMPASS_ENABLE': 0,
        'COMPASS_USE': 0,
        'COMPASS_USE2': 0,
        'COMPASS_USE3': 0,
        'ARMING_CHECK': 0,
        'FS_EKF_THRESH': 0.0
    }

    for param_id, param_value in params.items():
        boat_controller.set_param(param_id, param_value)

    # Set EKF home position
    home_latitude = 47.397742  # degrees
    home_longitude = 8.545594  # degrees
    home_altitude = 488.0  # meters
    boat_controller.set_ekf_home(home_latitude, home_longitude, home_altitude)

    # Start sending vision position estimates
    fake_x = 0.0  # meters
    fake_y = 0.0  # meters
    fake_z = 0.0  # meters
    fake_roll = 0.0  # radians
    fake_pitch = 0.0  # radians
    fake_yaw = 0.0  # radians
    boat_controller.start_vision_position_estimate(fake_x, fake_y, fake_z, fake_roll, fake_pitch, fake_yaw)

    try:
        # User menu for mode control
        while True:
            menu()
            choice = input("command code :: ")

            if choice == '1':
                current_mode = boat_controller.get_current_mode()
                print(f"Current mode is: {current_mode}")
            elif choice == '2':
                boat_controller.set_manual_mode()
            elif choice == '3':
                boat_controller.set_guided_mode()
            elif choice == '4':
                boat_controller.set_auto_mode()
            elif choice == '5':
                boat_controller.set_loiter_mode()
            elif choice == '6':
                boat_controller.arm_vehicle()
            elif choice == '7':
                boat_controller.disarm_vehicle()
            elif choice == '8':
                current_position = boat_controller.get_current_position()
                #print(f"Current position is: x={current_position[0]}, y={current_position[1]}, z={current_position[2]}, roll={current_position[3]}, pitch={current_position[4]}, yaw={current_position[5]}")
                print(f"Initial position: lat={current_position[0]}, lon={current_position[1]}")
            elif choice == '0':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
    except KeyboardInterrupt:
        print("Exiting...")

    # Stop sending vision position estimates and disconnect
    boat_controller.stop_vision_position_estimate()
    boat_controller.disconnect()
