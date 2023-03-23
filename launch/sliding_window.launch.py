import os
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription


def generate_launch_description():

    sliding_window_node = launch_ros.actions.Node(
        package='cc_visual_computing',
        executable='sliding_window',
        name='SlidingWindow',
        parameters=[os.path.join(get_package_share_directory('cc_visual_computing'), 'config', 'parameters.yaml')],
        output='screen',
    )

    return LaunchDescription([
        sliding_window_node,
    ])