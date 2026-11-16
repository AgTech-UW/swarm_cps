"""Rehearsal with nothing plugged in: simulated rovers and camera, the real
planner and foreman. Nothing is sent to any rover.

    ros2 launch swarm_cps rehearse.launch.py                               # mock model
    ros2 launch swarm_cps rehearse.launch.py backend:=ollama model:=qwen2.5:7b
    ros2 launch swarm_cps rehearse.launch.py decider:=rule

Then, in a second terminal:  ros2 run swarm_cps console   (type: go)
Watch /tmp/swarm_view.jpg, or add view:=window.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _s(text):
    """A parameter that stays a string whatever it looks like ("1", "true", "a:b")."""
    return ParameterValue(text, value_type=str)


def _get(context, name):
    return LaunchConfiguration(name).perform(context)


def _nodes(context):
    cfg = _get(context, 'config_dir')
    decider = _get(context, 'decider')
    common = {'config_dir': _s(cfg)} if cfg else {}
    out = [
        Node(package='swarm_cps', executable='sim', name='swarm_sim', output='screen',
             parameters=[common] if common else []),
        Node(package='swarm_cps', executable='planner', name='swarm_planner', output='screen',
             parameters=[dict(common, decider=_s(decider), robot_output=False, view=_s(_get(context, 'view')))]),
    ]
    if decider == 'foreman':
        out.append(Node(package='swarm_cps', executable='foreman', name='swarm_foreman', output='screen',
                        parameters=[{'backend': _s(_get(context, 'backend')), 'model': _s(_get(context, 'model')),
                                     'host': _s(_get(context, 'host'))}]))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_dir', default_value=''),
        DeclareLaunchArgument('decider', default_value='foreman'),
        DeclareLaunchArgument('view', default_value='file'),
        DeclareLaunchArgument('backend', default_value='mock'),
        DeclareLaunchArgument('model', default_value='qwen2.5:7b'),
        DeclareLaunchArgument('host', default_value='http://127.0.0.1:11434'),
        OpaqueFunction(function=_nodes),
    ])
