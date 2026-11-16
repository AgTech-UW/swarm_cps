"""Command relay for one rover. Start it with the rover's ROS_DOMAIN_ID in the
environment (the launch file does). It receives the planner's commands over
UDP on localhost and publishes them as geometry_msgs/Twist on the rover's
command topic, 20 times a second. If the planner goes quiet for 0.4 s it
publishes zero, and after two seconds of quiet it stops publishing so a
gamepad can drive the rover.

    ROS_DOMAIN_ID=5 ros2 run swarm_cps cmd_relay --ros-args -p robot:=rover1
"""
import json
import os
import socket
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .bridge import relay_port
from .settings import Settings


class CmdRelay(Node):
    def __init__(self):
        super().__init__('swarm_cmd_relay')
        self.declare_parameter('robot', '')
        self.declare_parameter('config_dir', '')
        name = self.get_parameter('robot').value
        st = Settings(self.get_parameter('config_dir').value or None)
        robot = next((r for r in st.robots if r['name'] == name), None)
        if robot is None:
            raise RuntimeError('cmd_relay: no robot named %r in fleet.yaml' % name)
        self.topic = robot.get('cmd_topic', '/cmd_vel')
        port = relay_port(st.robots, name, st.fleet.get('relay_port_base', 47600))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.setblocking(False)
        self.pub = self.create_publisher(Twist, self.topic, 10)
        self.v = self.w = 0.0
        self.t_last = -1e9
        self.warned = 0.0
        self.create_timer(0.05, self.tick)
        want = robot.get('domain_id')
        have = os.environ.get('ROS_DOMAIN_ID', '0')
        if want is not None and str(want) != str(have):
            self.get_logger().error('relay for %s is running on ROS_DOMAIN_ID=%s but fleet.yaml says %s'
                                    % (name, have, want))
        self.get_logger().info('relay for %s: UDP 127.0.0.1:%d -> %s on domain %s' % (name, port, self.topic, have))

    def tick(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except (BlockingIOError, OSError):
                break
            try:
                d = json.loads(data.decode())
                self.v, self.w = float(d['v']), float(d['w'])
                self.t_last = time.monotonic()
            except Exception:
                continue
        age = time.monotonic() - self.t_last
        if age > 2.0:
            return                                            # planner gone: leave the topic alone
        m = Twist()
        if age <= 0.4:
            m.linear.x, m.angular.z = self.v, self.w
        self.pub.publish(m)
        if self.pub.get_subscription_count() == 0 and time.monotonic() - self.warned > 5.0:
            self.warned = time.monotonic()
            self.get_logger().warn('nobody is subscribed to %s on this domain: is the rover on, booted, and on '
                                   'this ROS_DOMAIN_ID?' % self.topic)


def main(args=None):
    rclpy.init(args=args)
    node = CmdRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
