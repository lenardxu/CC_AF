import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription


def generate_launch_description():

    probabilistic_tracking_node = launch_ros.actions.Node(
        package='cc_visual_computing',
        executable='probabilistic_tracking',
        name='ProbabilisticTracking',
        parameters=[os.path.join(get_package_share_directory('cc_visual_computing'), 'config', 'parameters_probabilistic.yaml')],
        output='screen',
    )

    return LaunchDescription([
        probabilistic_tracking_node,
    ])

