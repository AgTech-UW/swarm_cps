"""The AI Foreman as a ROS node: observation in, decision out, model call in
between. It holds no geometry and steers nothing. The planner sends it the
system prompt and the observation; it returns the model's choice, which the
planner then checks and filters. If this node dies, the crew carries on with
the last accepted choice and the safety filter.

  in   /swarm/observation  std_msgs/String  {"id", "system", "observation"}
  out  /swarm/decision     std_msgs/String  {"id", "decision", "thought", "seconds", "raw"}

    ros2 run swarm_cps foreman --ros-args -p model:=qwen2.5:7b
    ros2 run swarm_cps foreman --ros-args -p backend:=mock          # no model needed
    ros2 run swarm_cps foreman --ros-args -p host:=http://192.168.0.105:11434 -p model:=qwen2.5:3b   # model on a rover's Pi
"""
import json
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .core.foreman import Foreman, MockModel, OllamaClient, OpenAIClient


class ForemanNode(Node):
    def __init__(self):
        super().__init__('swarm_foreman')
        self.declare_parameter('backend', 'ollama')          # ollama | openai | mock
        self.declare_parameter('model', 'qwen2.5:7b')
        self.declare_parameter('host', 'http://127.0.0.1:11434')
        self.declare_parameter('temperature', 0.3)
        self.declare_parameter('votes', 1)
        self.declare_parameter('timeout', 60.0)
        self.declare_parameter('log_dir', os.path.expanduser('~/swarm_logs'))
        g = lambda k: self.get_parameter(k).value            # noqa: E731
        backend = g('backend')
        if backend == 'mock':
            client = MockModel()
        elif backend == 'openai':
            client = OpenAIClient(g('model'), g('host'), float(g('temperature')), float(g('timeout')),
                                  os.environ.get('OPENAI_API_KEY'))
        else:
            client = OllamaClient(g('model'), g('host'), float(g('temperature')), float(g('timeout')))
        ok, msg = client.ping()
        (self.get_logger().info if ok else self.get_logger().error)(msg)
        self.fm = Foreman(client, votes=int(g('votes')))
        self.pub = self.create_publisher(String, '/swarm/decision', 10)
        self.create_subscription(String, '/swarm/observation', self.on_obs, 10)
        self.lock = threading.Lock()
        self.pending = None
        self.busy = False
        os.makedirs(g('log_dir'), exist_ok=True)
        self.path = os.path.join(g('log_dir'), time.strftime('foreman_%Y%m%d_%H%M%S.jsonl'))
        self.get_logger().info('foreman ready (%s). Transcript: %s' % (backend, self.path))

    def on_obs(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            return
        with self.lock:
            self.pending = d                                 # only the newest question matters
            if self.busy:
                return
            self.busy = True
        threading.Thread(target=self.work, daemon=True).start()

    def work(self):
        while True:
            with self.lock:
                d, self.pending = self.pending, None
                if d is None:
                    self.busy = False
                    return
            if d.get('system') and d['system'] != self.fm.system:
                self.fm.set_system(d['system'])              # new instruction: new conversation
            t0 = time.time()
            rep = self.fm.decide(d.get('observation', {}))
            sec = time.time() - t0
            raw = self.fm.transcript[-1]['replies'][0] if self.fm.transcript else ''
            out = dict(id=d.get('id', 0), decision=rep, thought=self.fm.last_thought if rep else '',
                       seconds=round(sec, 2), raw=raw[:2000])
            msg = String()
            msg.data = json.dumps(out)
            self.pub.publish(msg)
            self.get_logger().info('%.1f s -> %s' % (sec, json.dumps(rep)[:200] if rep else 'UNREADABLE: ' + raw[:120]))
            try:
                with open(self.path, 'a') as f:
                    f.write(json.dumps(dict(t=time.time(), observation=d.get('observation'), reply=raw,
                                            seconds=round(sec, 2))) + '\n')
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = ForemanNode()
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
