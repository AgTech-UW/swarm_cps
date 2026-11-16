"""Simulated rovers and a simulated camera, speaking the real topics. Run this
INSTEAD OF perception and the rovers to rehearse the whole ROS graph (planner,
foreman, console, logging) with nothing plugged in:

    ros2 launch swarm_cps rehearse.launch.py

It listens to each rover's intent topic /<name>/auto/cmd_vel, moves kinematic
Ackermann rovers with actuator lag and a 0.5 s motor watchdog, and publishes
/<name>/gps/pose and /<name>/gps/vel with camera-like noise and delay. "reset"
on /swarm/control puts the rovers back on their start marks.
"""
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .core.controller import SwarmController
from .core.sim import SimWorld
from .settings import Settings

POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


class SimNode(Node):
    def __init__(self):
        super().__init__('swarm_sim')
        self.declare_parameter('config_dir', '')
        self.st = Settings(self.get_parameter('config_dir').value or None)
        self.names = self.st.names
        self.ctl = SwarmController(self.st.arena(), self.names, self.st.params())
        self.make_world()
        self.cmd = {n: (0.0, 0.0) for n in self.names}
        for n in self.names:
            self.create_subscription(Twist, '/%s/auto/cmd_vel' % n, lambda m, n=n: self.on_cmd(n, m), 10)
        self.create_subscription(String, '/swarm/control', self.on_control, 10)
        self.pose_pub = {n: self.create_publisher(PoseStamped, '/%s/gps/pose' % n, POSE_QOS) for n in self.names}
        self.vel_pub = {n: self.create_publisher(TwistStamped, '/%s/gps/vel' % n, POSE_QOS) for n in self.names}
        self.t_prev = time.monotonic()
        self.last_stamp = {}
        self.create_timer(1.0 / 30.0, self.tick)
        self.get_logger().info('simulated rovers on their start marks: %s' % ', '.join(self.names))

    def make_world(self):
        self.world = SimWorld(self.names, self.ctl.start_poses(), true_turn_radius=self.ctl.p.min_turn_radius)

    def on_cmd(self, name, m):
        self.world.command({name: (m.linear.x, m.angular.z)})

    def on_control(self, m):
        if m.data.strip() == 'reset':
            self.make_world()
            self.get_logger().info('rovers back on their start marks')

    def tick(self):
        now = time.monotonic()
        dt = min(now - self.t_prev, 0.2)
        self.t_prev = now
        self.world.step(dt)
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        for n, s in self.world.measure().items():
            stamp = now_ros - s.age
            if stamp <= self.last_stamp.get(n, 0.0):
                continue                                      # no new frame for this rover
            self.last_stamp[n] = stamp
            sec = int(math.floor(stamp))
            ps = PoseStamped()
            ps.header.frame_id = 'arena'
            ps.header.stamp.sec, ps.header.stamp.nanosec = sec, int((stamp - sec) * 1e9)
            ps.pose.position.x, ps.pose.position.y = s.x, s.y
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(s.yaw / 2), math.cos(s.yaw / 2)
            self.pose_pub[n].publish(ps)
            tw = TwistStamped()
            tw.header = ps.header
            tw.twist.linear.x = s.v
            self.vel_pub[n].publish(tw)


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
