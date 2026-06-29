#!/bin/bash
# run_eval.sh — clean launch wrapper for the surround demo
#
# Kills all previous processes, then launches the surround demo with the
# supplied launch arguments.  Any ros2 launch arg works; the new ones are:
#
#   checkpoint_path:=<path>      Path to agent.th checkpoint
#   noise_std:=<float>           Range-noise fraction (0, 0.05, 0.1, 0.2)
#   target_moving:=<true|false>  Enable moving target
#   target_speed:=<float>        Target max_thrust (default 8.0)
#   max_thrust:=<float>          Hunter max_thrust (default 40.0)
#   heading_kp:=<float>          Hunter heading P-gain (default 8.0)
#   episode_length:=<int>        Auto-kill after N seconds (0 = never)
#
# Examples:
#   ./run_eval.sh
#   ./run_eval.sh checkpoint_path:=/path/to/agent.th noise_std:=0.1 target_moving:=true episode_length:=120
#   ./run_eval.sh noise_std:=0 target_moving:=false episode_length:=60

set -e

echo "=================================================="
echo "  Surround Demo – Clean Launch"
echo "=================================================="

# ---- 1. Kill everything from previous runs --------------------------
bash /home/blueboat_sitl/gz_ws/cleanup_demo.sh

# ---- 2. Source environment ------------------------------------------
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash
source ~/gz_ws/gazebo_exports.sh
source ~/gz_ws/install/setup.bash

# ---- 3. Launch with user-supplied args ------------------------------
echo ""
echo "Launch arguments: $*"
echo "=================================================="
echo ""

exec ros2 launch move_blueboat launch_surround_demo.launch.py "$@"
