from pymavlink import mavutil
import json
import time
import datetime
import logging
from logging.handlers import RotatingFileHandler

log_file_path = 'aggregated_telemetry.log'
logger = logging.getLogger('TelemetryLogger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

master = mavutil.mavlink_connection('udp:127.0.0.1:14550')

aggregated_data = {}

aggregation_interval = 1  
last_aggregation_time = time.time()

try:
    while True:
        msg = master.recv_match(blocking=True)
        if msg:
            msg_dict = msg.to_dict()
            
            message_type = msg.get_type()
            aggregated_data[message_type] = msg_dict
            
            current_time = time.time()
            if current_time - last_aggregation_time >= aggregation_interval:
                timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                aggregated_data["timestamp"] = timestamp
                
                aggregated_json = json.dumps(aggregated_data, indent=2)
                
                logger.info(aggregated_json)
                logger.info("==========================================")
                
                print(aggregated_json)
                print("==========================================")
                
                aggregated_data = {}
                
                last_aggregation_time = current_time
except KeyboardInterrupt:
    print("\nLogging terminated by user.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
