"""Pre-flight check of everything between the camera and the wheels. Run it
with the system up (perception or the simulator, and the rovers switched on):

    ros2 run swarm_cps doctor

It reads the configuration, listens for five seconds, and prints PASS or FAIL
for each link in the chain, with the likely cause next to every FAIL. It sends
nothing to the rovers.
"""
import json
import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from ..bridge import ContextBridge
from ..core.controller import SwarmController
from ..settings import Settings
from .common import say, verdict

POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


def main(argv=None):
    ok = True
    say('configuration')
    try:
        st = Settings(os.environ.get('SWARM_CONFIG_DIR'))
    except Exception as e:
        verdict(False, str(e))
        return 1
    verdict(True, 'config directory %s' % st.dir)
    try:
        arena, p = st.arena(), st.params()
        ctl = SwarmController(arena, st.names, p)
        probs = arena.validate(ctl.route, p.body_radius, p.min_turn_radius)
        ok &= verdict(not probs, 'arena.yaml: route %.1f m' % ctl.route.length if not probs else 'arena.yaml: ' + '; '.join(probs))
    except Exception as e:
        ok &= verdict(False, 'arena.yaml / fleet.yaml: %s' % e)
        return 1
    try:
        fm = st.floor_map()
        if fm is None:
            ok &= verdict(False, 'no camera_pose.yaml: run  ros2 run swarm_cps calibrate floor')
        else:
            rep = fm.info or {}
            ok &= verdict(True, 'floor calibration present (camera %.2f m up, fit %.1f px)'
                          % (fm.center[2], rep.get('reproj_rms_px', float('nan'))))
    except Exception as e:
        ok &= verdict(False, 'camera_pose.yaml: %s' % e)
    rmw = os.environ.get('RMW_IMPLEMENTATION', '(default)')
    dom = int(os.environ.get('ROS_DOMAIN_ID', '0') or 0)
    say('')
    say('ROS: RMW_IMPLEMENTATION=%s  ROS_DOMAIN_ID=%d' % (rmw, dom))
    clash = [r['name'] for r in st.robots if r.get('domain_id') is not None and int(r['domain_id']) == dom
             and r.get('cmd_topic', '/cmd_vel') == '/cmd_vel']
    if len(clash) > 1:
        ok &= verdict(False, '%s share this domain and all listen on /cmd_vel: one command would drive them all'
                      % ', '.join(clash))
    rclpy.init(args=argv)
    node = Node('swarm_doctor')
    stamps = {n: [] for n in st.names}
    lat = {n: [] for n in st.names}
    box = {}

    def on_pose(n, m):
        now = node.get_clock().now().nanoseconds * 1e-9
        s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        stamps[n].append(s)
        lat[n].append(now - s)
    for n in st.names:
        node.create_subscription(PoseStamped, '/%s/gps/pose' % n, lambda m, n=n: on_pose(n, m), POSE_QOS)
    node.create_subscription(String, '/swarm/perception', lambda m: box.__setitem__('perception', m.data), 10)
    node.create_subscription(String, '/swarm/status', lambda m: box.__setitem__('status', m.data), 10)
    bridge = None
    try:
        bridge = ContextBridge(st.robots, dom, log=lambda s: None)
    except Exception as e:
        ok &= verdict(False, 'cannot join the rover domains from one process (%s): set "bridge: relay" in fleet.yaml' % e)
    say('listening for 5 s ...')
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    say('')
    say('poses')
    for n in st.names:
        k = len(stamps[n])
        if k < 5:
            ok &= verdict(False, '%s: no poses on /%s/gps/pose (perception not running, tag %s not in view, or '
                          'wrong ROS_DOMAIN_ID in this terminal)' % (n, n, next(r['tag_id'] for r in st.robots if r['name'] == n)))
            continue
        rate = (k - 1) / max(stamps[n][-1] - stamps[n][0], 1e-6)
        age = sorted(lat[n])[len(lat[n]) // 2]
        good = rate > 12 and 0 <= age < 0.25
        ok &= verdict(good, '%s: %.0f poses/s, %.0f ms old on arrival%s' % (n, rate, 1e3 * age, '' if good else
                      ' (need > 12 /s and < 250 ms: see /swarm/perception for the cause)'))
    if 'perception' in box:
        h = json.loads(box['perception'])
        say('  perception reports: %s' % json.dumps({k: h[k] for k in ('rate_hz', 'detect_ms', 'camera_fps', 'late_frames',
                                                                       'skipped', 'anchor_error_m') if k in h}))
        if h.get('anchor_error_m') is not None:
            ok &= verdict(h['anchor_error_m'] < 0.03, 'floor reference tags are %.0f mm from their calibrated places'
                          % (1e3 * h['anchor_error_m']))
    say('')
    say('command path')
    for r in st.robots:
        n = r['name']
        topic = r.get('cmd_topic', '/cmd_vel')
        if bridge is not None and bridge.handles(n):
            c = bridge.listeners(n)
            ok &= verdict(c > 0, '%s: %d subscriber(s) on %s in domain %s%s' % (n, c, topic, r['domain_id'], '' if c > 0 else
                          '  <- rover off or still booting, wrong domain_id in fleet.yaml, or its controller is not running'))
        else:
            c = node.count_subscribers(topic)
            ok &= verdict(c > 0, '%s: %d subscriber(s) on %s in this domain' % (n, c, topic))
    say('')
    say('planner')
    if 'status' in box:
        s = json.loads(box['status'])
        verdict(True, 'planner is up: %s, phase %s, decider %s' % ('ARMED' if s.get('armed') else 'disarmed', s.get('phase'), s.get('decider')))
    else:
        say('  [ -- ] planner not running (fine if you have not started it yet)')
    if bridge is not None:
        bridge.close()
    node.destroy_node()
    rclpy.shutdown()
    say('')
    say('RESULT: %s' % ('ready' if ok else 'not ready: fix the FAIL lines, top to bottom'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
