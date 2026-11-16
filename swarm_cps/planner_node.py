"""The swarm planner: poses in, per-rover speed and yaw-rate intent out.

  in   /<name>/gps/pose, /<name>/gps/vel     from perception (or the simulator)
       /swarm/control      std_msgs/String    "arm" | "stop" | "reset" | "decider <name>"
       /swarm/heartbeat    std_msgs/String    from the operator console; lost for 1 s = stop
       /swarm/instruction  std_msgs/String    the operator's sentence
       /swarm/decision     std_msgs/String    the foreman's reply (JSON)
  out  /<name>/auto/cmd_vel geometry_msgs/Twist  always, in this domain (the paper's intent topic)
       the rover's own command topic, through the bridge (see fleet.yaml)
       /swarm/observation  std_msgs/String    what the foreman is asked (JSON, with the system prompt)
       /swarm/status       std_msgs/String    5 Hz JSON: phase, formation, stops, ages

The node starts disarmed and commands nothing until the console says "arm".
Every cycle it uses the newest pose of every rover and refuses to move the
fleet if any is older than the stale timeout, judged both by the stamp on the
pose and by when it arrived here.

    ros2 run swarm_cps planner --ros-args -p decider:=rule
    ros2 run swarm_cps planner --ros-args -p decider:=foreman
"""
import csv
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .bridge import ContextBridge, UdpBridge
from .core.controller import RobotState, SwarmController
from .core.deciders import FixedColumn, Rule, Schedule
from .core.foreman import build_system
from .core.sim import Metrics
from .settings import Settings

POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=1)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
DECIDERS = dict(column=FixedColumn, schedule=Schedule, rule=Rule)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Planner(Node):
    def __init__(self):
        super().__init__('swarm_planner')
        self.declare_parameter('config_dir', '')
        self.declare_parameter('decider', 'column')
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('require_heartbeat', True)
        self.declare_parameter('view', 'file')               # off | file | window
        self.declare_parameter('view_path', '/tmp/swarm_view.jpg')
        self.declare_parameter('log_dir', os.path.expanduser('~/swarm_logs'))
        self.declare_parameter('robot_output', True)         # False = dry run: intent topics only
        log = self.get_logger()
        self.st = Settings(self.get_parameter('config_dir').value or None)
        self.arena = self.st.arena()
        self.params = self.st.params()
        self.names = self.st.names
        self.ctl = SwarmController(self.arena, self.names, self.params)
        self.metrics = Metrics(self.arena, self.ctl.route, self.params.body_radius)
        for prob in self.arena.validate(self.ctl.route, self.params.body_radius, self.params.min_turn_radius):
            log.error('arena.yaml: %s' % prob)
        if self.params.spacing <= self.params.protected:
            log.warn('spacing (%.2f) is not above protected range (%.2f): the formation repels itself (Part I)'
                     % (self.params.spacing, self.params.protected))
        self.rate = float(self.get_parameter('rate').value)
        self.need_hb = bool(self.get_parameter('require_heartbeat').value)
        self.view = self.get_parameter('view').value
        self.view_path = self.get_parameter('view_path').value
        self.instruction = ''
        self.decider_name = ''
        self.decider = None
        self.thought = ''
        self.set_decider(self.get_parameter('decider').value)
        # state from the camera
        self.pose = {}                                        # name -> (x, y, yaw, stamp_s, t_arrival_mono)
        self.vel = {}
        for n in self.names:
            self.create_subscription(PoseStamped, '/%s/gps/pose' % n, lambda m, n=n: self.on_pose(n, m), POSE_QOS)
            self.create_subscription(TwistStamped, '/%s/gps/vel' % n, lambda m, n=n: self.on_vel(n, m), POSE_QOS)
        self.create_subscription(String, '/swarm/control', self.on_control, 10)
        self.create_subscription(String, '/swarm/heartbeat', self.on_heartbeat, 10)
        self.create_subscription(String, '/swarm/instruction', self.on_instruction, LATCHED)
        self.create_subscription(String, '/swarm/decision', self.on_decision, 10)
        self.intent = {n: self.create_publisher(Twist, '/%s/auto/cmd_vel' % n, 10) for n in self.names}
        self.obs_pub = self.create_publisher(String, '/swarm/observation', 10)
        self.status_pub = self.create_publisher(String, '/swarm/status', 10)
        # ways out to the rovers
        self.own_domain = int(os.environ.get('ROS_DOMAIN_ID', '0') or 0)
        self.direct = {}
        self.bridge = None
        if bool(self.get_parameter('robot_output').value):
            mode = str(self.st.fleet.get('bridge', 'context'))
            if mode == 'relay':
                self.bridge = UdpBridge(self.st.robots, self.own_domain, self.st.fleet.get('relay_port_base', 47600))
            else:
                try:
                    self.bridge = ContextBridge(self.st.robots, self.own_domain, log=lambda s: log.info(str(s)))
                except Exception as e:
                    log.error('could not open the other ROS domains (%s). Set "bridge: relay" in fleet.yaml and '
                              'start with the launch file, which runs one relay per rover.' % e)
            for r in self.st.robots:
                topic = r.get('cmd_topic', '/cmd_vel')
                same = r.get('domain_id') is None or int(r['domain_id']) == self.own_domain
                if same and topic != '/%s/auto/cmd_vel' % r['name']:
                    self.direct[r['name']] = self.create_publisher(Twist, topic, 10)
        else:
            log.warn('robot_output is false: DRY RUN, nothing is sent to the rovers')
        # per-rover output calibration: what to multiply a command by so the rover does what was asked
        self.scale = {r['name']: (float(r.get('v_scale', 1.0)), float(r.get('w_scale', 1.0))) for r in self.st.robots}
        self.t_hb = -1e9
        self.t_prev = time.monotonic()
        self.t_disarm = -1e9
        self.was_armed = False
        self.obs_id = 0
        self.t_status = 0.0
        self.t_view = 0.0
        self.td = None
        self.csv = None
        self.run_dir = None
        self.log_dir = self.get_parameter('log_dir').value
        self.n_tick = 0
        self.warned_listeners = 0.0
        self.create_timer(1.0 / self.rate, self.tick)
        sp = self.ctl.start_poses()
        log.info('planner ready, DISARMED. decider=%s. Put the rovers here (x, y, heading):' % self.decider_name)
        for n in self.names:
            log.info('  %-8s (%.2f, %.2f) facing %s' % (n, sp[n][0], sp[n][1], self.arena.compass(sp[n][2])))

    # ---- inputs ----
    def on_pose(self, name, m):
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        old = self.pose.get(name)
        if old is not None and stamp <= old[3]:
            return                                            # repeated or reordered: not a new observation
        if m.header.frame_id not in ('arena', ''):
            self.get_logger().warn('pose for %s is in frame %r, not arena; ignored' % (name, m.header.frame_id),
                                   throttle_duration_sec=5.0)
            return
        self.pose[name] = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation), stamp, time.monotonic())

    def on_vel(self, name, m):
        self.vel[name] = m.twist.linear.x

    def on_heartbeat(self, m):
        self.t_hb = time.monotonic()

    def on_instruction(self, m):
        text = m.data.strip()
        if text == self.instruction:
            return
        self.instruction = text
        self.get_logger().info('instruction: "%s"' % text)
        if hasattr(self.decider, 'begin'):
            self.decider.begin(self.ctl, text)
        self.ctl._trig['last'] = -1e9                         # ask the decider again now

    def set_decider(self, name):
        name = (name or 'column').strip().lower()
        if name in DECIDERS:
            self.decider = DECIDERS[name]()
            self.decider.begin(self.ctl, self.instruction)
        elif name == 'foreman':
            self.decider = None                               # decisions arrive on /swarm/decision
        else:
            self.get_logger().error('unknown decider %r (use column, schedule, rule or foreman)' % name)
            return
        self.decider_name = name
        self.thought = ''

    def on_control(self, m):
        cmd = m.data.strip()
        log = self.get_logger()
        if cmd == 'arm':
            missing = [n for n in self.names if n not in self.pose]
            if missing:
                log.error('not arming: no pose yet for %s' % ', '.join(missing))
                return
            if self.need_hb and time.monotonic() - self.t_hb > 1.0:
                log.error('not arming: no heartbeat from the console')
                return
            sp = self.ctl.start_poses()
            if not self.ctl.s:                                # a fresh run: are they where they should be?
                for n in self.names:
                    d = math.hypot(self.pose[n][0] - sp[n][0], self.pose[n][1] - sp[n][1])
                    if d > 0.35:
                        log.warn('%s is %.2f m from its start mark (%.2f, %.2f)' % (n, d, sp[n][0], sp[n][1]))
            ok, msg = self.ctl.arm()
            self.ctl._trig['last'] = -1e9                     # consult the decider as the run starts
            log.info(msg) if ok else log.error(msg)
            if ok and self.csv is None:
                self.open_log()
        elif cmd == 'stop':
            self.ctl.disarm()
            log.info('STOP')
        elif cmd == 'reset':
            self.finish_log()
            self.ctl.reset()
            self.metrics = Metrics(self.arena, self.ctl.route, self.params.body_radius)
            if hasattr(self.decider, 'begin'):
                self.decider.begin(self.ctl, self.instruction)
            self.td = None
            log.info('reset: disarmed, mission back to the start. Put the rovers back on their marks.')
        elif cmd.startswith('decider'):
            self.set_decider(cmd.split(None, 1)[1] if ' ' in cmd else '')
            log.info('decider is now %s' % self.decider_name)
        else:
            log.warn('unknown control command %r' % cmd)

    def on_decision(self, m):
        if self.decider_name != 'foreman':
            return
        try:
            d = json.loads(m.data)
        except Exception:
            return
        if d.get('id', self.obs_id) < self.obs_id - 1:
            return                                            # an answer to a question two decisions old
        states = self.states()
        dec = d.get('decision')
        rec = self.ctl.request(dec if dec is not None else 'unreadable reply', states, source='foreman')
        self.thought = str(d.get('thought', ''))[:300]
        if 'error' in rec:
            self.get_logger().warn('foreman reply not usable: %s' % rec['error'])
        else:
            self.get_logger().info('foreman: %s%s | %s' % (rec['applied'], ' (asked %s, held)' % rec['held']
                                                           if 'held' in rec else '', self.thought))
        self.log_decision(rec)

    # ---- the cycle ----
    def states(self):
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        now = time.monotonic()
        out = {}
        for n, (x, y, yaw, stamp, t_arr) in self.pose.items():
            age = max(now_ros - stamp, now - t_arr)           # old by its stamp OR by its arrival is old
            if now_ros - stamp < -0.25:
                age = float('inf')                            # a stamp from the future: clocks disagree
            out[n] = RobotState(x, y, yaw, self.vel.get(n, 0.0), age)
        return out

    def tick(self):
        now = time.monotonic()
        dt = now - self.t_prev
        self.t_prev = now
        log = self.get_logger()
        self.n_tick += 1
        if self.need_hb and self.ctl.armed and now - self.t_hb > 1.0:
            self.ctl.disarm()
            log.error('console heartbeat lost: STOP')
        states = self.states()
        if dt > 0.5:                                          # we were stalled: do not act on that interval
            cmds = {n: (0.0, 0.0) for n in self.names}
            self.ctl.step(states, 0.0)
        else:
            why = self.ctl.decision_due(states) if all(n in states for n in self.names) else []
            if why:
                self.ask(states, why)
            cmds = self.ctl.step(states, dt)
        if self.ctl.armed != self.was_armed:
            if not self.ctl.armed:
                self.t_disarm = now
                if self.ctl.done:
                    log.info('MISSION COMPLETE')
                    self.finish_log()
                elif self.ctl.latched:
                    log.error('FLEET STOP: %s' % self.ctl.latched)
            self.was_armed = self.ctl.armed
        if self.ctl.armed and self.ctl.fleet_stop:
            log.warn('holding: %s' % self.ctl.fleet_stop, throttle_duration_sec=2.0)
        # send. Disarmed: zeros for two seconds, then silence so a gamepad can drive.
        active = self.ctl.armed or (now - self.t_disarm < 2.0)
        for n in self.names:
            v, w = cmds.get(n, (0.0, 0.0)) if self.ctl.armed else (0.0, 0.0)
            if active:
                m = Twist()
                m.linear.x, m.angular.z = float(v), float(w)
                self.intent[n].publish(m)
                vs, ws = self.scale[n]
                if n in self.direct:
                    m2 = Twist()
                    m2.linear.x, m2.angular.z = float(v * vs), float(w * ws)
                    self.direct[n].publish(m2)
                if self.bridge is not None and self.bridge.handles(n):
                    self.bridge.send(n, v * vs, w * ws)
        if self.ctl.armed and all(n in states for n in self.names):
            pos = {n: (states[n].x, states[n].y) for n in self.names}
            working = self.ctl.working_names()
            self.metrics.update(pos, min(dt, 0.5), working, list(self.ctl.stop_cause))
            if self.csv is not None and self.n_tick % 3 == 0:
                self.write_row(states, cmds)
        if now - self.t_status > 0.2:
            self.t_status = now
            self.publish_status(states)
        if self.view != 'off' and now - self.t_view > (0.2 if self.view == 'window' else 0.5):
            self.t_view = now
            self.draw(states)
        if self.bridge is not None and self.ctl.armed and now - self.warned_listeners > 5.0:
            self.warned_listeners = now
            deaf = [n for n in self.names if self.bridge.handles(n) and self.bridge.listeners(n) == 0]
            if deaf:
                log.warn('nothing is subscribed to the command topic of: %s (rover off, wrong domain_id, or '
                         'wrong cmd_topic in fleet.yaml)' % ', '.join(deaf))

    def ask(self, states, why):
        ob = self.ctl.observation(states, why, self.instruction, self.metrics)
        if self.decider_name == 'foreman':
            self.obs_id += 1
            m = String()
            m.data = json.dumps(dict(id=self.obs_id, system=build_system(self.ctl, self.instruction), observation=ob))
            self.obs_pub.publish(m)
        elif self.decider is not None:
            rep = self.decider.decide(ob)
            if rep is not None:
                rec = self.ctl.request(rep, states, source=self.decider_name)
                self.thought = '%s: %s' % (self.decider_name, rec.get('applied', rec.get('error')))
                self.log_decision(rec)

    # ---- outputs for people ----
    def publish_status(self, states):
        st = dict(self.ctl.status)
        st.update(decider=self.decider_name, instruction=self.instruction, thought=self.thought,
                  latched=self.ctl.latched, metrics=self.metrics.result() if self.metrics.t > 0 else None,
                  poses={n: [round(s.x, 3), round(s.y, 3), round(s.yaw, 3)] for n, s in states.items()},
                  heartbeat_ok=(time.monotonic() - self.t_hb) < 1.0)
        st['cmd'] = {n: list(c) for n, c in st.get('cmd', {}).items()}
        m = String()
        m.data = json.dumps(st)
        self.status_pub.publish(m)

    def draw(self, states):
        try:
            import cv2
            from .core.viz import TopDown
            if self.td is None:
                self.td = TopDown(self.arena)
            poses = {n: (s.x, s.y, s.yaw) for n, s in states.items()}
            img = self.td.draw(self.ctl, poses, self.metrics, title='ARMED' if self.ctl.armed else 'disarmed',
                               thought=self.thought, instruction=self.instruction,
                               extra=['Decider: %s' % self.decider_name])
            if self.view == 'window':
                cv2.imshow('swarm', img)
                cv2.waitKey(1)
            else:
                tmp = self.view_path + '.tmp.jpg'
                cv2.imwrite(tmp, img)
                os.replace(tmp, self.view_path)
        except Exception as e:
            self.get_logger().warn('view disabled (%s)' % e)
            self.view = 'off'

    def open_log(self):
        try:
            self.run_dir = os.path.join(self.log_dir, time.strftime('run_%Y%m%d_%H%M%S'))
            os.makedirs(self.run_dir, exist_ok=True)
            self._csv_file = open(os.path.join(self.run_dir, 'trajectory.csv'), 'w', newline='')
            self.csv = csv.writer(self._csv_file)
            self.csv.writerow(['t', 'robot', 'x', 'y', 'yaw', 'v_meas', 'pose_age', 'cmd_v', 'cmd_w', 'phase',
                               'formation', 'offset', 'stop'])
            with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
                json.dump(dict(arena=self.arena.cfg, fleet=self.st.fleet, decider=self.decider_name,
                               instruction=self.instruction, start=time.time()), f, indent=1)
            self.get_logger().info('logging to %s' % self.run_dir)
        except Exception as e:
            self.get_logger().warn('no log (%s)' % e)
            self.csv = None

    def write_row(self, states, cmds):
        st = self.ctl.status
        for n in self.names:
            s = states[n]
            c = cmds.get(n, (0.0, 0.0))
            self.csv.writerow(['%.3f' % self.ctl.t, n, '%.4f' % s.x, '%.4f' % s.y, '%.4f' % s.yaw, '%.3f' % s.v,
                               '%.3f' % s.age, '%.3f' % c[0], '%.3f' % c[1], st.get('phase', ''), self.ctl.formation,
                               '%.3f' % self.ctl.offset_applied, self.ctl.stop_cause.get(n, self.ctl.fleet_stop or '')])

    def log_decision(self, rec):
        if self.run_dir is None:
            return
        try:
            with open(os.path.join(self.run_dir, 'decisions.jsonl'), 'a') as f:
                f.write(json.dumps(dict(rec, thought=self.thought, instruction=self.instruction)) + '\n')
        except Exception:
            pass

    def finish_log(self):
        if self.run_dir is None:
            return
        try:
            res = self.metrics.result()
            res.update(completed=bool(self.ctl.done), decider=self.decider_name, instruction=self.instruction,
                       filter_log=[r for r in self.ctl.log if 'asked' not in r])
            with open(os.path.join(self.run_dir, 'result.json'), 'w') as f:
                json.dump(res, f, indent=1)
            self._csv_file.close()
            self.get_logger().info('result: %s' % json.dumps({k: res[k] for k in (
                'completed', 'time_s', 'collisions', 'min_clearance_m', 'zone_machine_seconds', 'field_worked_pct')}))
            self.get_logger().info('saved %s' % self.run_dir)
        except Exception as e:
            self.get_logger().warn('could not save the result (%s)' % e)
        self.csv = None
        self.run_dir = None

    def close(self):
        z = Twist()
        for _ in range(3):                                    # leave the rovers stopped
            for n in self.names:
                try:
                    self.intent[n].publish(z)
                    if n in self.direct:
                        self.direct[n].publish(z)
                    if self.bridge is not None and self.bridge.handles(n):
                        self.bridge.send(n, 0.0, 0.0)
                except Exception:
                    pass
            time.sleep(0.05)
        self.finish_log()
        if self.bridge is not None:
            self.bridge.close()


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
