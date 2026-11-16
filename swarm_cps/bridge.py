"""Getting velocity commands to rovers that live on other ROS domains.

In the course set-up every rover is on its own ROS_DOMAIN_ID and listens on
/cmd_vel, which is what keeps one group's commands away from another group's
rover. A swarm planner has to talk to all of them. Two ways to do that without
touching the rovers:

ContextBridge  one extra rclpy context (DDS participant) per rover domain,
               inside the planner process. rclpy.init(context=..., domain_id=...)
               is the supported way to have several domains in one process.
UdpBridge      the planner sends each command to a small relay process
               (swarm_cps cmd_relay) that was started with that rover's
               ROS_DOMAIN_ID; the relay publishes /cmd_vel there. Same result
               with only the most ordinary ROS usage, for when the first way
               misbehaves on some middleware build.
"""
import json
import socket
import time


class ContextBridge:
    def __init__(self, robots, own_domain, log=print):
        import rclpy
        from geometry_msgs.msg import Twist
        self._Twist = Twist
        self._rclpy = rclpy
        self.ctx = {}
        self.nodes = {}
        self.pubs = {}
        self.topic = {}
        for r in robots:
            d = r.get('domain_id')
            if d is None or int(d) == int(own_domain):
                continue
            d = int(d)
            if d not in self.ctx:
                ctx = rclpy.Context()
                # args=[] and use_global_arguments=False: this participant must not
                # inherit the planner's command line (node name remap, parameter file)
                rclpy.init(args=[], context=ctx, domain_id=d)
                self.ctx[d] = ctx
                self.nodes[d] = rclpy.create_node('swarm_cmd_bridge_%d' % d, context=ctx, use_global_arguments=False)
                log('command bridge: joined ROS domain %d' % d)
            self.topic[r['name']] = r.get('cmd_topic', '/cmd_vel')
            self.pubs[r['name']] = self.nodes[d].create_publisher(Twist, self.topic[r['name']], 10)

    def handles(self, name):
        return name in self.pubs

    def send(self, name, v, w):
        m = self._Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pubs[name].publish(m)

    def listeners(self, name):
        """How many subscribers the rover's command topic has (0 = nobody is listening)."""
        try:
            return int(self.pubs[name].get_subscription_count())
        except Exception:
            return -1

    def close(self):
        for d, node in self.nodes.items():
            try:
                node.destroy_node()
            except Exception:
                pass
        for d, ctx in self.ctx.items():
            try:
                self._rclpy.shutdown(context=ctx)
            except Exception:
                pass


class UdpBridge:
    def __init__(self, robots, own_domain, port_base=47600, log=print):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = {}
        for i, r in enumerate(robots):
            d = r.get('domain_id')
            if d is None or int(d) == int(own_domain):
                continue
            self.port[r['name']] = int(port_base) + i
        self.seq = 0

    def handles(self, name):
        return name in self.port

    def send(self, name, v, w):
        self.seq += 1
        msg = json.dumps(dict(v=float(v), w=float(w), seq=self.seq, t=time.time())).encode()
        try:
            self.sock.sendto(msg, ('127.0.0.1', self.port[name]))
        except OSError:
            pass

    def listeners(self, name):
        return -1                                             # the relay reports that itself

    def close(self):
        self.sock.close()


def relay_port(robots, name, port_base=47600):
    for i, r in enumerate(robots):
        if r['name'] == name:
            return int(port_base) + i
    raise KeyError(name)
