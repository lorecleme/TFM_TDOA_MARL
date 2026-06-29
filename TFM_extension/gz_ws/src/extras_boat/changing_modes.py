from pymavlink import mavutil
import time

class BoatController:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.conn = None

    def connect(self):
        # Create the connection
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
        self.connect()
        # Try to arm the vehicle
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0)

        # Wait and check for arming confirmation
        print("Attempting to arm the vehicle...")
        timeout = time.time() + 10  # 10 second timeout for arming
        armed = False

        while not armed and time.time() < timeout:
            message = self.conn.recv_match(type='HEARTBEAT', blocking=True)
            if message:
                print(message.to_dict())
                armed = message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                if armed:
                    print('Vehicle is armed!')
                    break

        if not armed:
            print('Failed to arm the vehicle. Check pre-arm conditions and messages.')

        self.disconnect()

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

        # Print status after setting mode
        self.print_status(f"after setting to {mode}")

        self.disconnect()

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
        self.connect()
        """Get the current mode of the vehicle"""
        message = self.conn.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        current_mode = None
        if message:
            custom_mode = message.custom_mode
            for mode, mode_id in self.conn.mode_mapping().items():
                if mode_id == custom_mode:
                    current_mode = mode
                    break
        self.disconnect()
        return current_mode

    def print_status(self, description):
        print(f"Current status {description}:")
        message = self.conn.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        if message:
            print(message.to_dict())
        else:
            print("No heartbeat message received.")

def menu():
    print("\nBoat Controller Menu")
    print("1. Check current mode")
    print("2. Set mode to MANUAL")
    print("3. Set mode to GUIDED")
    print("4. Set mode to AUTO")
    print("5. Set mode to LOITER")
    print("6. Arm vehicle")
    print("0. Exit")

if __name__ == "__main__":
    boat_controller = BoatController('udpin:0.0.0.0:14550')
    #boat_controller = BoatController('udpin:192.168.2.1:14550')
    
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
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
