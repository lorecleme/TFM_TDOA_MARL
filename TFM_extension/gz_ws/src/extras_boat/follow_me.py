# Markus Buchholz
# Follow me (GPS)
# https://ardupilot.org/rover/docs/follow-mode.
# https://github.com/mavlink/qgroundcontrol/issues/7811

"""
To make this work, the ground station must publish it's position at 1Hz (or faster if possible) to the vehicle using the GLOBAL_POSITION_INT message.
Beyond this it's possible that the GCS user could modify the position of the vehicle etc with one of the existing mavlink parameters.

I can not set vehicle into FOLLOW mode. We use GUIDED!

MAVPROXY:
mode
mode FOLLOW
"""

from pymavlink import mavutil
import time
import math

# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

# Wait for the heartbeat message to find the system ID
master.wait_heartbeat()

def arm_vehicle():
    # Try to arm the vehicle
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0)

    # Wait and check for arming confirmation
    print("Attempting to arm the vehicle...")
    timeout = time.time() + 10  # 10 second timeout for arming
    armed = False

    while not armed and time.time() < timeout:
        message = master.recv_match(type='HEARTBEAT', blocking=True)
        if message:
            print(message.to_dict())
            armed = message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if armed:
                print('Vehicle is armed!')
                break

    if not armed:
        print('Failed to arm the vehicle. Check pre-arm conditions and messages.')

def set_guided_mode():
    mode = 'GUIDED'
    if mode not in master.mode_mapping():
        print(f"Mode {mode} not found in mode mapping. Exiting...")
        return False
    
    mode_id = master.mode_mapping()[mode]
    print(f"Setting mode to GUIDED with mode_id {mode_id}")

    # Ensure vehicle is disarmed before setting mode
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0, 0, 0, 0, 0, 0, 0)
    print("Disarming the vehicle to change mode.")
    time.sleep(2)

    timeout = time.time() + 15  # 15 second timeout for setting mode
    while time.time() < timeout:
        master.set_mode(mode_id)
        ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
        if ack:
            if ack.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print("Mode set to GUIDED.")
                return True
            elif ack.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE and ack.result == mavutil.mavlink.MAV_RESULT_FAILED:
                print("Failed to set mode to GUIDED. Trying again...")
        time.sleep(1)

    print("Timeout while setting mode to GUIDED.")
    return False

def get_initial_position():
    print("Waiting for GPS position...")
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            initial_lat = msg.lat / 1e7
            initial_lon = msg.lon / 1e7
            print(f"Initial position: lat={initial_lat}, lon={initial_lon}")
            return initial_lat, initial_lon

def send_position_target(lat, lon, alt=0):
    # Send the position target to the vehicle
    master.mav.set_position_target_global_int_send(
        0,                         # time_boot_ms
        master.target_system,      # target_system
        master.target_component,   # target_component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111111000,        # type_mask (only positions enabled)
        int(lat * 1e7),            # lat_int - X Position in WGS84 frame in 1e7 * degrees
        int(lon * 1e7),            # lon_int - Y Position in WGS84 frame in 1e7 * degrees
        int(alt * 1000),           # alt
        0, 0, 0,                   # X, Y, Z velocity in m/s (not used)
        0, 0, 0,                   # afx, afy, afz acceleration (not used)
        0, 0)                      # yaw, yaw rate (not used)

def follow_me(initial_lat, initial_lon):
    # Convert meters to degrees
    meter_to_lat = 1 / 111320
    meter_to_lon = 1 / (111320 * math.cos(math.radians(initial_lat)))

    # Target position 10 meters north and 10 meters east of the initial position
    target_lat = initial_lat + 10 * meter_to_lat
    target_lon = initial_lon + 10 * meter_to_lon
    
    current_lat = initial_lat
    current_lon = initial_lon
    
    # Number of steps to move to the target
    steps = 100
    lat_step = (target_lat - initial_lat) / steps
    lon_step = (target_lon - initial_lon) / steps

    # Move from initial position to (10, 10)
    for _ in range(steps):
        current_lat += lat_step
        current_lon += lon_step
        send_position_target(current_lat, current_lon)
        print(f"Sent position target: lat={current_lat}, lon={current_lon}")
        time.sleep(0.1)  # send at 5 Hz

    # Move back from (10, 10) to initial position
    for _ in range(steps):
        current_lat -= lat_step
        current_lon -= lon_step
        send_position_target(current_lat, current_lon)
        print(f"Sent position target: lat={current_lat}, lon={current_lon}")
        time.sleep(0.1)  # send at 1 Hz

if __name__ == "__main__":
    if set_guided_mode():
        arm_vehicle()
        initial_lat, initial_lon = get_initial_position()
        follow_me(initial_lat, initial_lon)
