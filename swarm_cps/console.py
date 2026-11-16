"""Operator console. Run it in its own terminal (not from a launch file: it
reads the keyboard).

    ros2 run swarm_cps console

  go                 arm: the crew starts (or carries on)
  <Enter>            STOP. An empty line is the fastest thing to type.
  stop               STOP
  reset              back to the start of the mission (put the rovers back first)
  say <sentence>     give the operator's instruction, e.g.
                       say Neighbour's got cattle along the north fence. Keep the machines well back from that side.
  decider <name>     column | schedule | rule | foreman
  status             one-line status
  quit

While it runs it sends a heartbeat; if this terminal closes or the laptop
freezes, the planner stops the fleet within a second. Ctrl+C here also stops.
"""
import json
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


class Console(Node):
    def __init__(self):
        super().__init__('swarm_console')
        self.ctrl = self.create_publisher(String, '/swarm/control', 10)
        self.hb = self.create_publisher(String, '/swarm/heartbeat', 10)
        self.ins = self.create_publisher(String, '/swarm/instruction', LATCHED)
        self.create_subscription(String, '/swarm/status', self.on_status, 10)
        self.status = None
        self.last_line = ''
        self.create_timer(0.2, self.beat)

    def beat(self):
        m = String()
        m.data = 'alive'
        self.hb.publish(m)

    def send(self, pub, text):
        m = String()
        m.data = text
        pub.publish(m)

    def on_status(self, m):
        try:
            self.status = json.loads(m.data)
        except Exception:
            return
        line = self.summary(short=True)
        if line != self.last_line:                            # only say something when something changed
            self.last_line = line
            print('  [' + line + ']', flush=True)

    def summary(self, short=False):
        s = self.status
        if not s:
            return 'no status from the planner yet'
        parts = ['ARMED' if s.get('armed') else ('DONE' if s.get('done') else 'disarmed'), 'phase %s' % s.get('phase'),
                 '%s' % s.get('formation'), 'offset %+.2f' % s.get('offset_target', 0.0)]
        if s.get('held'):
            parts.append('holding back %s' % s['held'])
        if s.get('fleet_stop'):
            parts.append('FLEET STOP: %s' % s['fleet_stop'])
        for n, why in (s.get('stops') or {}).items():
            parts.append('%s %s' % (n, why))
        if not short:
            parts.append('speed %.2f' % s.get('speed', 0.0))
            parts.append('pose age %s' % s.get('age'))
            if s.get('metrics'):
                parts.append('zones %s' % s['metrics'].get('zone_machine_seconds'))
            if s.get('thought'):
                parts.append('decider: %s' % s['thought'])
        return ' | '.join(parts)


def main(args=None):
    rclpy.init(args=args)
    node = Console()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    print(__doc__)
    try:
        while True:
            line = sys.stdin.readline()
            if line == '':
                break                                        # terminal closed
            cmd = line.strip()
            low = cmd.lower()
            if low in ('', 'stop', 's', 'x'):
                node.send(node.ctrl, 'stop')
                print('  STOP sent')
            elif low in ('go', 'arm', 'g'):
                node.send(node.ctrl, 'arm')
            elif low == 'reset':
                node.send(node.ctrl, 'reset')
            elif low.startswith('decider'):
                node.send(node.ctrl, low)
            elif low == 'status':
                print('  ' + node.summary())
            elif low in ('quit', 'exit', 'q'):
                break
            elif low.startswith('say '):
                node.send(node.ins, cmd[4:].strip())
                print('  instruction sent')
            else:
                print('  not a command. To give an instruction: say <sentence>. Enter alone = STOP.')
    except KeyboardInterrupt:
        pass
    finally:
        node.send(node.ctrl, 'stop')
        print('  STOP sent, console closing')
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
