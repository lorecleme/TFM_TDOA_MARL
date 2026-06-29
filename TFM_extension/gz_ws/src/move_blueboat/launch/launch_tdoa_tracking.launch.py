#!/usr/bin/env python3
"""
Launch file for TDOA-based multi-agent tracking with TransfQMix.

Starts:
  1. Gazebo with the multi-vessel waves world
  2. 3 ArduPilot SITL instances (one per hunter, unique FDM ports)
  3. ROS-Gazebo bridges for all 3 hunters + target vessel
  4. TDOA observation node
  5. TransfQMix inference node
  6. Action mapper node (discrete → MAVLink RC_CHANNELS_OVERRIDE)
  7. Target vessel waypoint controller

Usage:
  ros2 launch move_blueboat launch_tdoa_tracking.launch.py \
    checkpoint_path:=/path/to/agent.th
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='/home/blueboat_sitl/gz_ws/models/working_models/turbo_normal_moving_450000/450000/agent.th',
        description='Path to agent.th checkpoint'
    )

    checkpoint_path = LaunchConfiguration('checkpoint_path')

    hunter_vessels = ['blueboat_1', 'blueboat_2', 'blueboat_3']
    target_vessel = 'target_vessel'

    # Bridge topics for all 4 vessels
    bridge_topics = []
    for vessel in hunter_vessels:
        bridge_topics.extend([
            f'/model/{vessel}/joint/motor_port_joint/cmd_thrust'
            f'@std_msgs/msg/Float64@ignition.msgs.Double',
            f'/model/{vessel}/joint/motor_stbd_joint/cmd_thrust'
            f'@std_msgs/msg/Float64@ignition.msgs.Double',
            f'/model/{vessel}/odometry'
            f'@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
        ])
    bridge_topics.append(
        f'/model/{target_vessel}/odometry'
        f'@nav_msgs/msg/Odometry@ignition.msgs.Odometry'
    )
    # Top-down camera for video recording
    bridge_topics.append(
        '/top_down_camera@sensor_msgs/msg/Image@ignition.msgs.Image'
    )

    # Base location (Rio de Janeiro, matching waves.sdf spherical coordinates)
    HOME_LAT = '-22.986687'
    HOME_LON = '-43.202501'
    HOME_ALT = '0'
    HOME_HDG = '0'

    return LaunchDescription([
        checkpoint_arg,

        # ── Gazebo (server + GUI in split mode for smoother playback) ──
        ExecuteProcess(
            cmd=['gz', 'sim', '-s', 'waves.sdf'],
            output='log',
            name='gz_server',
        ),
        # Launch GUI after server is ready
        TimerAction(
            period=8.0,
            actions=[
                ExecuteProcess(
                    cmd=['gz', 'sim', '-g'],
                    output='screen',
                    name='gz_gui',
                ),
            ],
        ),

        # ── 3 ArduPilot SITL instances (one per hunter boat) ──
        # FDM ports in model.sdf: blueboat_1=5501, blueboat_2=5511, blueboat_3=5521
        # SITL instances match: -I0→5501, -I1→5511, -I2→5521
        # --no-mavproxy skips the MAVProxy GUI (avoids numpy 2.x errors)
        ExecuteProcess(
            cmd=['/home/blueboat_sitl/ardupilot/Tools/autotest/sim_vehicle.py', '-v', 'Rover', '-f', 'gazebo-rover',
                 '--model', 'JSON', '-I0',
                 '--sim-address=127.0.0.1:5501',
                 '-l', f'{HOME_LAT},{HOME_LON},{HOME_ALT},{HOME_HDG}',
                 '--out=udpout:127.0.0.1:14550'],
            output='screen',
            name='sitl_boat1',
        ),
        ExecuteProcess(
            cmd=['/home/blueboat_sitl/ardupilot/Tools/autotest/sim_vehicle.py', '-v', 'Rover', '-f', 'gazebo-rover',
                 '--model', 'JSON', '-I1',
                 '--sim-address=127.0.0.1:5511',
                 '-l', f'{HOME_LAT},{HOME_LON},{HOME_ALT},{HOME_HDG}',
                 '--out=udpout:127.0.0.1:14560'],
            output='screen',
            name='sitl_boat2',
        ),
        ExecuteProcess(
            cmd=['/home/blueboat_sitl/ardupilot/Tools/autotest/sim_vehicle.py', '-v', 'Rover', '-f', 'gazebo-rover',
                 '--model', 'JSON', '-I2',
                 '--sim-address=127.0.0.1:5521',
                 '-l', f'{HOME_LAT},{HOME_LON},{HOME_ALT},{HOME_HDG}',
                 '--out=udpout:127.0.0.1:14570'],
            output='screen',
            name='sitl_boat3',
        ),

        # ── ROS-Gazebo bridge ──
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=bridge_topics,
            output='log',
            name='tdoa_ros_gz_bridge',
        ),

        # ── TDOA Observation Node ──
        Node(
            package='move_blueboat',
            executable='tdoa_observation_node',
            output='screen',
            name='tdoa_observation_node',
            parameters=[{'noise_std': 0.05, 'world_span': 1000.0, 'rate_hz': 10.0,
                         'landmark_source': 'ground_truth'}],
        ),

        # ── TransfQMix Inference Node ──
        Node(
            package='move_blueboat',
            executable='transf_qmix_inference_node',
            output='screen',
            name='transf_qmix_inference_node',
            parameters=[{'checkpoint_path': checkpoint_path}],
        ),

        # ── Action Mapper Node (MAVLink RC_CHANNELS_OVERRIDE) ──
        Node(
            package='move_blueboat',
            executable='tdoa_action_mapper_node',
            output='screen',
            name='tdoa_action_mapper_node',
            parameters=[{'mavlink_ports': [14550, 14560, 14570],
                         'max_thrust': 40.0, 'heading_kp': 3.0, 'rate_hz': 10.0}],
        ),

        # ── Target Vessel Controller ──
        Node(
            package='move_blueboat',
            executable='target_waypoint_controller',
            output='screen',
            name='target_waypoint_controller',
        ),
    ])
