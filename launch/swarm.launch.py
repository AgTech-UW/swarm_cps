"""The real thing: camera -> poses -> planner -> rovers.

    ros2 launch swarm_cps swarm.launch.py                          # single file, no model
    ros2 launch swarm_cps swarm.launch.py decider:=rule
    ros2 launch swarm_cps swarm.launch.py decider:=foreman model:=qwen2.5:7b
    ros2 launch swarm_cps swarm.launch.py decider:=foreman backend:=mock
    ros2 launch swarm_cps swarm.launch.py robot_output:=false      # watch what it WOULD command

The planner starts disarmed. Arm it from a second terminal:
    ros2 run swarm_cps console

The rovers' commands are published to each rover's own domain (fleet.yaml).
If fleet.yaml says "bridge: relay", this file also starts one relay per rover.
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
    common = {'config_dir': _s(cfg)} if cfg else {}        # empty = let the node find the config itself
    p_perc = dict(common, source=_s(_get(context, 'source')), show=_get(context, 'show').lower() == 'true')
    p_plan = dict(common, decider=_s(decider), view=_s(_get(context, 'view')),
                  robot_output=_get(context, 'robot_output').lower() == 'true')
    out = [
        Node(package='swarm_cps', executable='perception', name='swarm_perception', output='screen', parameters=[p_perc]),
        Node(package='swarm_cps', executable='planner', name='swarm_planner', output='screen', parameters=[p_plan]),
    ]
    if decider == 'foreman':
        out.append(Node(package='swarm_cps', executable='foreman', name='swarm_foreman', output='screen',
                        parameters=[{'backend': _s(_get(context, 'backend')), 'model': _s(_get(context, 'model')),
                                     'host': _s(_get(context, 'host'))}]))
    try:
        from swarm_cps.settings import Settings
        st = Settings(cfg or None)
        if str(st.fleet.get('bridge', 'context')) == 'relay':
            for r in st.robots:
                if r.get('domain_id') is None:
                    continue
                out.append(Node(package='swarm_cps', executable='cmd_relay', name='swarm_cmd_relay_' + r['name'],
                                output='screen', parameters=[dict(common, robot=_s(r['name']))],
                                additional_env={'ROS_DOMAIN_ID': str(int(r['domain_id']))}))
    except Exception as e:
        print('[swarm.launch] could not read fleet.yaml for relays: %s' % e)
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_dir', default_value='', description='folder with arena.yaml, fleet.yaml, camera.yaml'),
        DeclareLaunchArgument('decider', default_value='column', description='column | schedule | rule | foreman'),
        DeclareLaunchArgument('source', default_value='v4l2', description='v4l2 | ros_compressed | apriltag_ros'),
        DeclareLaunchArgument('show', default_value='false', description='open a window with the camera image'),
        DeclareLaunchArgument('view', default_value='file', description='top-down view: off | file | window'),
        DeclareLaunchArgument('robot_output', default_value='true', description='false = dry run'),
        DeclareLaunchArgument('backend', default_value='ollama', description='foreman: ollama | openai | mock'),
        DeclareLaunchArgument('model', default_value='qwen2.5:7b'),
        DeclareLaunchArgument('host', default_value='http://127.0.0.1:11434'),
        OpaqueFunction(function=_nodes),
    ])
