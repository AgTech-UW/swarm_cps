from glob import glob

from setuptools import setup

package_name = 'swarm_cps'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.core', package_name + '.tools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AgTech-UW',
    maintainer_email='agtech@example.com',
    description='Corner-camera faux GPS, swarm planner and LLM foreman for small Ackermann rovers.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception = swarm_cps.perception_node:main',
            'planner = swarm_cps.planner_node:main',
            'foreman = swarm_cps.foreman_node:main',
            'console = swarm_cps.console:main',
            'sim = swarm_cps.sim_node:main',
            'cmd_relay = swarm_cps.cmd_relay:main',
            'doctor = swarm_cps.tools.doctor:main',
            'calibrate = swarm_cps.tools.calibrate:main',
            'cam_bench = swarm_cps.tools.cam_bench:main',
            'plan_mount = swarm_cps.tools.plan_mount:main',
            'layout = swarm_cps.tools.layout:main',
            'simulate = swarm_cps.tools.simulate:main',
            'tag_sheet = swarm_cps.tools.tag_sheet:main',
            'draw_path = swarm_cps.tools.draw_path:main',
        ],
    },
)
