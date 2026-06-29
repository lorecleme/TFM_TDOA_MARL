from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    # Generate bridge topics for each vessel
    hunter_vessels = ['blueboat_1', 'blueboat_2', 'blueboat_3']
    target_vessel = 'target_vessel'

    bridge_topics = []

    # Bridge per-hunter topics (thrust + odometry)
    for vessel in hunter_vessels:
        bridge_topics.extend([
            f'/model/{vessel}/joint/motor_port_joint/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double',
            f'/model/{vessel}/joint/motor_stbd_joint/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double',
            f'/model/{vessel}/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
        ])

    # Bridge target vessel odometry
    bridge_topics.extend([
        f'/model/{target_vessel}/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
    ])

    # Shared sensors (camera + lidar, from original blueboat if present)
    bridge_topics.extend([
        '/navsat@sensor_msgs/msg/NavSatFix@ignition.msgs.NavSat',
        '/camera@sensor_msgs/msg/Image@ignition.msgs.Image',
        '/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
        '/laser_scan@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan',
    ])

    return LaunchDescription([
        # Launch Gazebo simulation
        ExecuteProcess(
            cmd=['env', '__GLX_VENDOR_LIBRARY_NAME=nvidia', 'gz', 'sim', 'waves.sdf'],
            output='screen'
        ),

        # Launch the Gazebo-ROS bridge for multi-vessel TDOA tracking
        Node(
                package='ros_ign_bridge',
            executable='parameter_bridge',
            arguments=bridge_topics,
            output='screen'
        ),
    ])

