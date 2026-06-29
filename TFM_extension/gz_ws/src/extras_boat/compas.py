# Markus Buchholz

from pymavlink import mavutil
import time


master = mavutil.mavlink_connection('udpin:192.168.2.1:14550')
master.wait_heartbeat()

def read_compass_heading(master):
    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        1, 1
    )

    print("Waiting for compass heading data...")
    while True:
        message = master.recv_match(type='VFR_HUD', blocking=True)
        if message:
            heading = message.heading
            # Normalize heading to 0-360 degrees
            heading_normalized = heading % 360
            print(f"Compass heading: {heading_normalized} degrees")
            return heading_normalized

# read compass heading
heading = read_compass_heading(master)

print(f"Normalized compass heading (0 degrees is North): {heading} degrees")
