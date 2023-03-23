import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription


def generate_launch_description():

    return LaunchDescription([
        #launch_ros.actions.Node(
        #    package='cc_visual_computing',
        #    executable='image_preprocessing',
        #    name='image_preprocessing',
        #    output='screen'
        #),
        #launch_ros.actions.Node(
        #    package='cc_visual_computing',
        #    executable='lane_detection',
        #    name='lane_detection',
        #    output='screen'
        #),
        #launch_ros.actions.Node(
        #    package='cc_visual_computing',
        #    executable='sliding_window',
        #    name='sliding_window',
        #    parameters=[os.path.join(get_package_share_directory('cc_visual_computing'), 'config', 'parameters.yaml')],
        #    output='screen'
        #),
        launch_ros.actions.Node(
            package='cc_visual_computing',
            executable='probabilistic_tracking',
            name='ProbabilisticTracking',
            parameters=[os.path.join(get_package_share_directory('cc_visual_computing'), 'config', 'parameters_probabilistic.yaml')],
            output='screen',
        ),
    ])
