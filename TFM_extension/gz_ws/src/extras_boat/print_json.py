from pymavlink import mavutil
import json
import time

master = mavutil.mavlink_connection('udp:127.0.0.1:14550')

aggregated_data = {}

aggregation_interval = 1  # Adjust as needed
last_aggregation_time = time.time()

while True:
    msg = master.recv_match(blocking=True)
    if msg:
        msg_dict = msg.to_dict()
        
        message_type = msg.get_type()
        aggregated_data[message_type] = msg_dict
        
        current_time = time.time()
        if current_time - last_aggregation_time >= aggregation_interval:
            print(json.dumps(aggregated_data, indent=2))
            
            print("==========================================")
            
            aggregated_data = {}
            
            last_aggregation_time = current_time
