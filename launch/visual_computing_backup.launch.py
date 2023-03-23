"""Start the visual computing for Carolo-Cup."""

import os
from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import LogInfo

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg=['Start visual computing for the Carolo-Cup']),
        DeclareLaunchArgument('sw', default_value='false',
                              description='true == execute sliding window'),
        DeclareLaunchArgument('pt', default_value='false',
                              description='true == execute probabilistic tracking'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('cc_visual_computing'),
                'launch/sliding_window.launch.py')),
            condition=IfCondition(LaunchConfiguration('sw'))
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('cc_visual_computing'),
                'launch/probabilistic_tracking.launch.py')),
            condition=IfCondition(LaunchConfiguration('pt'))
        ),
    ])
