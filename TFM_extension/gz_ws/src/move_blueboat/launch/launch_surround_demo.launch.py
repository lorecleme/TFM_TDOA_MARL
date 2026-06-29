#!/usr/bin/env python3
"""
Launch file for the TransfQMix 3v1 surround demo (NO ArduPilot SITL).

Starts:
  1. Gazebo with the minimal waves.sdf world
     (3 hunters clustered ~45 m south of a target at origin)
  2. ROS-Gazebo bridge: cmd_thrust (port+stbd) and odometry for ALL 4 vessels
  3. tdoa_observation_node     (ROS odom -> 72-float entity obs, 10 Hz)
  4. transf_qmix_inference_node (loads agent.th, greedy actions, 10 Hz)
  5. tdoa_action_mapper_node   (actions -> world-frame heading PD -> thrust)
  6. target_waypoint_controller (conditionally, if target_moving:=true)
  7. surround_metrics_node      (computes & prints surround quality to stdout)

The hunters are driven by direct Gazebo Thruster-plugin thrust commands (no
SITL, no MAVLink) so the policy's bang-bang control is not filtered by an
autopilot guidance layer — this reproduces the trained MPE surround behaviour.

Experiment settings (all exposed as launch args):
  checkpoint_path       – path to trained TransfQMix agent.th
  landmark_source       – 'ground_truth' (old checkpoints) or 'tdoa_estimate' (B1)
  noise_std             – range-noise fraction of world_span (0 / 0.05 / 0.1 / 0.2)
  target_moving         – true: launch target waypoint controller
  target_speed          – target controller max_thrust (default 8.0)
  max_thrust            – hunter max differential thrust (default 40.0)
  heading_kp            – hunter heading P-gain (default 8.0)
  episode_length        – auto-kill metrics node after N seconds (0 = never)

Usage (Mode A – old checkpoint, works out of the box):
  ros2 launch move_blueboat launch_surround_demo.launch.py \\
    checkpoint_path:=/path/to/turbo_normal_moving_450000/agent.th \\
    landmark_source:=ground_truth

Usage (Mode B – B1 checkpoint, requires retraining with confidence gating):
  ros2 launch move_blueboat launch_surround_demo.launch.py \\
    checkpoint_path:=/path/to/b1_retrained/agent.th \\
    landmark_source:=tdoa_estimate \\
    noise_std:=0.05 \\
    target_moving:=true \\
    episode_length:=120

Or use the convenience wrapper:
  ./run_eval.sh checkpoint_path:=/path/to/agent.th noise_std:=0.1 target_moving:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # ---- launch arguments --------------------------------------------------
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='/home/blueboat_sitl/gz_ws/models/working_models/agent.th',
        description='Path to trained TransfQMix agent.th checkpoint')

    noise_arg = DeclareLaunchArgument(
        'noise_std',
        default_value='0.05',
        description='Range-noise fraction of world_span '
                    '(0=none, 0.05=50m, 0.1=100m, 0.2=200m)')

    landmark_source_arg = DeclareLaunchArgument(
        'landmark_source',
        default_value='ground_truth',
        description='Landmark-row position source: ground_truth (old checkpoints) '
                    'or tdoa_estimate (B1 checkpoints)')

    target_moving_arg = DeclareLaunchArgument(
        'target_moving',
        default_value='false',
        description='true = launch target waypoint controller')

    target_speed_arg = DeclareLaunchArgument(
        'target_speed',
        default_value='8.0',
        description='Target controller max_thrust')

    max_thrust_arg = DeclareLaunchArgument(
        'max_thrust',
        default_value='40.0',
        description='Hunter max differential thrust per motor')

    heading_kp_arg = DeclareLaunchArgument(
        'heading_kp',
        default_value='8.0',
        description='Hunter heading PD proportional gain')

    episode_length_arg = DeclareLaunchArgument(
        'episode_length',
        default_value='0',
        description='Auto-kill metrics node after N seconds (0 = never)')

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Run Gazebo headless (server-only, no GUI). Set true for automated eval.')

    seed_arg = DeclareLaunchArgument(
        'seed',
        default_value='0',
        description='Random seed for spawn positions (0 = time-based). Set for reproducible eval.')

    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value='waves.sdf',
        description='Gazebo world file: waves.sdf (256m, default) or waves_1024m.sdf (1024m)')

    world_span_arg = DeclareLaunchArgument(
        'world_span',
        default_value='256.0',
        description='World span for observation normalization (256 for 256m patch, 1000 for 1024m)')

    min_sep_arg = DeclareLaunchArgument(
        'min_sep',
        default_value='60.0',
        description='Minimum spawn distance from target (60m for 256m patch, 300m for 1024m)')

    baseline_scale_arg = DeclareLaunchArgument(
        'baseline_scale',
        default_value='1.0',
        description='Inflate agent-agent distances in obs (1.0=real, 3.0=training-scale baselines)')

    hidden_reset_arg = DeclareLaunchArgument(
        'hidden_reset_interval',
        default_value='1',
        description='Steps between hidden state reset (1=every step, 100=training episode length)')

    # ---- constants ---------------------------------------------------------
    hunter_vessels = ['blueboat_1', 'blueboat_2', 'blueboat_3']
    target_vessel = 'target_vessel'

    # Bridge cmd_thrust (both motors) + odometry for every vessel (4 vessels)
    bridge_topics = []
    for vessel in hunter_vessels + [target_vessel]:
        bridge_topics.extend([
            f'/model/{vessel}/joint/motor_port_joint/cmd_thrust'
            f'@std_msgs/msg/Float64@ignition.msgs.Double',
            f'/model/{vessel}/joint/motor_stbd_joint/cmd_thrust'
            f'@std_msgs/msg/Float64@ignition.msgs.Double',
            f'/model/{vessel}/odometry'
            f'@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
        ])

    return LaunchDescription([
        checkpoint_arg,
        noise_arg,
        landmark_source_arg,
        target_moving_arg,
        target_speed_arg,
        max_thrust_arg,
        heading_kp_arg,
        episode_length_arg,
        headless_arg,
        seed_arg,
        world_file_arg,
        world_span_arg,
        min_sep_arg,
        baseline_scale_arg,
        hidden_reset_arg,

        # ---- 0. Cleanup (optional, uncomment to kill old instances) ---------
        # ExecuteProcess(cmd=['bash', '/home/blueboat_sitl/gz_ws/cleanup_demo.sh'],
        #                output='screen', name='cleanup'),

        # ---- 1. Gazebo (headless or GUI, identical physics stack) -----------
        # Headless mode sets GZ_SIM_HEADLESS_RENDERING=1 (EGL offscreen).
        # GUI mode omits it (normal X11 window).  Both run the identical
        # physics + Ogre2 rendering pipeline — only the output target differs.
        # This ensures headless eval results are identical to supervised runs.
        ExecuteProcess(
            cmd=['env',
                 '__GLX_VENDOR_LIBRARY_NAME=nvidia',
                 'gz', 'sim', '-s', '-r', LaunchConfiguration('world_file')],
            output='screen',
            name='gz_sim_headless',
            condition=IfCondition(
                PythonExpression([
                    '"', LaunchConfiguration('headless'), '" == "true"'
                ])
            ),
        ),
        ExecuteProcess(
            cmd=['env',
                 '__GLX_VENDOR_LIBRARY_NAME=nvidia',
                 'gz', 'sim', '-r', LaunchConfiguration('world_file')],
            output='screen',
            name='gz_sim_gui',
            condition=UnlessCondition(
                PythonExpression([
                    '"', LaunchConfiguration('headless'), '" == "true"'
                ])
            ),
        ),

        # ---- 2. ROS-Gazebo bridge -------------------------------------------
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=bridge_topics,
            output='log',
            name='surround_ros_gz_bridge',
        ),

        # ---- 2b. Randomize spawn to match MPE training distribution -----------
        Node(
            package='move_blueboat',
            executable='randomize_spawn',
            output='screen',
            name='randomize_spawn',
            parameters=[{
                'min_sep': LaunchConfiguration('min_sep'),
                'world_span': LaunchConfiguration('world_span'),
                'seed': LaunchConfiguration('seed'),
            }],
        ),

        # ---- 3. Target controller (conditional) -----------------------------
        Node(
            package='move_blueboat',
            executable='target_waypoint_controller',
            output='screen',
            name='target_waypoint_controller',
            parameters=[{
                'max_thrust': LaunchConfiguration('target_speed'),
                'rate_hz': 10.0,
            }],
            condition=IfCondition(
                PythonExpression([
                    '"', LaunchConfiguration('target_moving'), '" == "true"'
                ])
            ),
        ),

        # ---- 4. TDOA Observation Node ---------------------------------------
        Node(
            package='move_blueboat',
            executable='tdoa_observation_node',
            output='screen',
            name='tdoa_observation_node',
            parameters=[{
                'noise_std': LaunchConfiguration('noise_std'),
                'world_span': LaunchConfiguration('world_span'),
                'rate_hz': 10.0,
                'landmark_source': LaunchConfiguration('landmark_source'),
                'baseline_scale': LaunchConfiguration('baseline_scale'),
            }],
        ),

        # ---- 5. TransfQMix Inference Node -----------------------------------
        Node(
            package='move_blueboat',
            executable='transf_qmix_inference_node',
            output='screen',
            name='transf_qmix_inference_node',
            parameters=[{
                'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                'hidden_reset_interval': LaunchConfiguration('hidden_reset_interval'),
            }],
        ),

        # ---- 6. Action Mapper Node (direct thrust, no SITL) -----------------
        Node(
            package='move_blueboat',
            executable='tdoa_action_mapper_node',
            output='screen',
            name='tdoa_action_mapper_node',
            parameters=[{
                'max_thrust': LaunchConfiguration('max_thrust'),
                'heading_kp': LaunchConfiguration('heading_kp'),
                'rate_hz': 10.0,
            }],
        ),

        # ---- 7. Metrics Node ------------------------------------------------
        Node(
            package='move_blueboat',
            executable='surround_metrics_node',
            output='screen',
            name='surround_metrics_node',
            parameters=[{
                'print_interval': 5.0,
                'episode_length': LaunchConfiguration('episode_length'),
            }],
        ),
    ])
