"""The ROS nodes, run against a strict stand-in for rclpy (test/fake_ros) so
that node code is executed, not just imported: constructors, callbacks,
timers, message field types, QoS matching, parameters, and the multi-domain
command bridge. What this cannot check is the real middleware and the real
camera; the runbook's first gates cover those."""
import importlib
import json
import math
import os
import shutil
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, 'fake_ros')
MODS = ['swarm_cps.bridge', 'swarm_cps.cmd_relay', 'swarm_cps.console', 'swarm_cps.foreman_node',
        'swarm_cps.perception_node', 'swarm_cps.planner_node', 'swarm_cps.sim_node', 'swarm_cps.tools.doctor']
CATTLE = "Neighbour's got cattle along the north fence. Keep the machines well back from that side."


@pytest.fixture()
def ros(tmp_path, monkeypatch):
    """Fresh fake ROS, fake time inside the node modules, and a private copy
    of the configuration with a floor calibration in it."""
    monkeypatch.syspath_prepend(FAKE)
    for m in [k for k in sys.modules if k.split('.')[0] in ('rclpy', 'geometry_msgs', 'std_msgs', 'sensor_msgs',
                                                             'builtin_interfaces', 'msgbase')]:
        del sys.modules[m]
    for m in MODS:
        sys.modules.pop(m, None)
    import rclpy
    from rclpy.clock import FakeTimeModule
    from rclpy.node import Node
    rclpy.reset()
    cfg = tmp_path / 'cfg'
    shutil.copytree(os.path.join(HERE, '..', 'config'), str(cfg))
    monkeypatch.setenv('SWARM_CONFIG_DIR', str(cfg))
    monkeypatch.setenv('ROS_DOMAIN_ID', '0')
    from swarm_cps.core.camera import FloorMap
    from swarm_cps.settings import Settings
    st = Settings()
    fm = FloorMap.from_mount(st.camera_model(), (-0.1, -0.1, 2.3), (0.63, 1.12, 0.0))
    st.save_floor_map(fm)
    mods = {}
    for m in MODS:
        mods[m.split('.')[-1]] = importlib.import_module(m)
    for mod in mods.values():
        if hasattr(mod, 'time'):
            monkeypatch.setattr(mod, 'time', FakeTimeModule)
    logs = tmp_path / 'logs'
    Node.OVERRIDES = {'swarm_planner': dict(view='off', log_dir=str(logs)), 'swarm_foreman': dict(log_dir=str(logs))}

    class Env:
        pass
    env = Env()
    env.rclpy, env.Node, env.mods, env.cfg, env.logs, env.st, env.fm = rclpy, Node, mods, str(cfg), str(logs), st, fm
    yield env
    Node.OVERRIDES = {}
    rclpy.reset()


def operator(env, heartbeat=True):
    """A stand-in for the console: heartbeat, and a way to send commands."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    node = env.Node('test_operator')
    hb = node.create_publisher(String, '/swarm/heartbeat', 10)
    ctrl = node.create_publisher(String, '/swarm/control', 10)
    ins = node.create_publisher(String, '/swarm/instruction', QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST, depth=1))
    node.beating = heartbeat
    node.create_timer(0.2, lambda: hb.publish(String(data='alive')) if node.beating else None)
    node.control = lambda text: ctrl.publish(String(data=text))
    node.say = lambda text: ins.publish(String(data=text))
    return node


def test_rehearsal_graph_runs_a_whole_mission_and_logs_it(ros):
    ros.Node.OVERRIDES['swarm_planner'].update(decider='rule', robot_output=False)
    ros.rclpy.init()
    sim = ros.mods['sim_node'].SimNode()
    planner = ros.mods['planner_node'].Planner()
    op = operator(ros)
    ros.rclpy.run_for(1.0)
    assert not planner.ctl.armed
    assert all(abs(v) < 1e-9 for s in sim.world.state.values() for v in s[3:])      # nothing moves before "arm"
    op.control('arm')
    assert planner.ctl.armed, [r for r in planner.get_logger().records if r[0] == 'error']
    ros.rclpy.run_for(90.0)
    assert planner.ctl.done, planner.ctl.status
    assert planner.metrics.result()['collisions'] == 0
    assert not [r for r in planner.get_logger().records if r[0] == 'error']
    runs = [d for d in os.listdir(ros.logs) if d.startswith('run_')]
    assert len(runs) == 1
    res = json.load(open(os.path.join(ros.logs, runs[0], 'result.json')))
    assert res['completed'] and res['decider'] == 'rule'
    rows = open(os.path.join(ros.logs, runs[0], 'trajectory.csv')).read().splitlines()
    assert rows[0].startswith('t,robot,x,y,yaw') and len(rows) > 500
    assert os.path.getsize(os.path.join(ros.logs, runs[0], 'decisions.jsonl')) > 0
    # and after the run the planner is silent on the intent topics (zeros for 2 s, then nothing)
    n = len(planner.intent['rover1'].sent)
    ros.rclpy.run_for(5.0)
    assert len(planner.intent['rover1'].sent) - n < 70


def test_losing_the_console_heartbeat_stops_the_fleet(ros):
    ros.Node.OVERRIDES['swarm_planner'].update(decider='column', robot_output=False)
    ros.rclpy.init()
    sim = ros.mods['sim_node'].SimNode()
    planner = ros.mods['planner_node'].Planner()
    op = operator(ros)
    ros.rclpy.run_for(1.0)
    op.control('arm')
    ros.rclpy.run_for(5.0)
    assert max(s[3] for s in sim.world.state.values()) > 0.05
    op.beating = False                                          # the console terminal is closed
    ros.rclpy.run_for(3.0)
    assert not planner.ctl.armed
    assert max(s[3] for s in sim.world.state.values()) < 0.005
    last = planner.intent['rover1'].sent[-1]
    assert last.linear.x == 0.0 and last.angular.z == 0.0
    # arming without a heartbeat is refused
    op.control('arm')
    assert not planner.ctl.armed
    # Enter (stop) and reset work
    op.beating = True
    ros.rclpy.run_for(0.5)
    op.control('arm')
    assert planner.ctl.armed
    op.control('stop')
    assert not planner.ctl.armed
    op.control('reset')
    assert planner.ctl.s == {} and not planner.ctl.done
    assert math.hypot(sim.world.state['rover1'][0] - planner.ctl.start_poses()['rover1'][0],
                      sim.world.state['rover1'][1] - planner.ctl.start_poses()['rover1'][1]) < 1e-6


def test_sentence_to_wheels_through_the_foreman_node(ros):
    """Instruction (latched, sent before the planner exists) -> planner ->
    observation -> foreman node -> decision -> safety filter -> commands."""
    ros.Node.OVERRIDES['swarm_planner'].update(decider='foreman', robot_output=False)
    ros.Node.OVERRIDES['swarm_foreman'].update(backend='mock')
    ros.rclpy.init()
    op = operator(ros)
    op.say(CATTLE)
    sim = ros.mods['sim_node'].SimNode()
    planner = ros.mods['planner_node'].Planner()
    foreman = ros.mods['foreman_node'].ForemanNode()
    assert planner.instruction == CATTLE                        # the late joiner got the latched instruction
    ros.rclpy.run_for(1.0)
    op.control('arm')
    import time as real_time
    for _ in range(900):
        ros.rclpy.run_for(0.1)
        while foreman.busy:                                     # let the model thread answer
            real_time.sleep(0.001)
        if planner.ctl.done:
            break
    assert planner.ctl.done, planner.ctl.status
    res = planner.metrics.result()
    assert res['collisions'] == 0
    assert res['zone_machine_seconds']['north strip'] < 1.0, res
    applied = [r for r in planner.ctl.log if r.get('source') == 'foreman' and 'applied' in r]
    assert len(applied) >= 3
    assert any(r['applied']['offset'] < -0.05 for r in applied)                    # it moved the passes south
    assert CATTLE in foreman.fm.system
    assert os.path.getsize(foreman.path) > 0
    assert sim is not None


def test_each_rover_gets_its_own_commands_in_its_own_domain(ros):
    from geometry_msgs.msg import Twist
    ros.Node.OVERRIDES['swarm_planner'].update(decider='column', robot_output=True)
    ros.rclpy.init()
    got = {}
    rovers = {}
    for r in ros.st.robots:                                     # each rover: its own domain, listening on /cmd_vel
        ctx = ros.rclpy.Context()
        ros.rclpy.init(context=ctx, domain_id=r['domain_id'])
        node = ros.rclpy.create_node('rover', context=ctx)
        got[r['name']] = []
        node.create_subscription(Twist, '/cmd_vel', lambda m, n=r['name']: got[n].append((m.linear.x, m.angular.z)), 10)
        rovers[r['name']] = node
    sim = ros.mods['sim_node'].SimNode()
    planner = ros.mods['planner_node'].Planner()
    assert planner.bridge is not None and all(planner.bridge.listeners(n) == 1 for n in ros.st.names)
    op = operator(ros)
    ros.rclpy.run_for(1.0)
    assert all(len(v) == 0 for v in got.values())               # disarmed and never armed: silent
    op.control('arm')
    ros.rclpy.run_for(6.0)
    for n in ros.st.names:
        assert len(got[n]) > 100, n
        assert all(isinstance(v, float) and isinstance(w, float) for v, w in got[n])
        assert max(v for v, _ in got[n]) > 0.05
    assert got['rover1'] != got['rover2'] != got['rover3']      # not one command stream copied to all
    for n in ros.st.names:                                      # and what each rover hears is its own intent
        sent = [(m.linear.x, m.angular.z) for m in planner.intent[n].sent]
        assert got[n] == sent[-len(got[n]):]
    # a rover that drops off the network is reported
    rovers['rover2'].destroy_node()
    ros.rclpy.run_for(6.0)
    assert any('rover2' in msg and 'nothing is subscribed' in msg for lvl, msg in planner.get_logger().records)
    planner.close()
    assert got['rover1'][-1] == (0.0, 0.0)                      # shutting down leaves a zero command
    assert sim is not None


def test_relay_bridge_carries_commands_over_udp(ros, monkeypatch):
    from geometry_msgs.msg import Twist
    import yaml
    path = os.path.join(ros.cfg, 'fleet.yaml')
    fleet = yaml.safe_load(open(path))
    fleet['bridge'] = 'relay'
    fleet['relay_port_base'] = 47890
    yaml.safe_dump(fleet, open(path, 'w'))
    monkeypatch.setenv('ROS_DOMAIN_ID', '2')                    # what the launch file sets for rover2's relay
    ros.Node.OVERRIDES['swarm_cmd_relay'] = dict(robot='rover2')
    ros.rclpy.init()
    relay = ros.mods['cmd_relay'].CmdRelay()
    heard = []
    rover = ros.Node('rover2_base')
    rover.create_subscription(Twist, '/cmd_vel', lambda m: heard.append((m.linear.x, m.angular.z)), 10)
    from swarm_cps.settings import Settings
    bridge = ros.mods['bridge'].UdpBridge(Settings().robots, own_domain=0, port_base=47890)
    assert bridge.handles('rover2')
    ros.rclpy.run_for(0.3)
    assert heard == []                                          # no planner yet: the relay stays off the topic
    bridge.send('rover2', 0.12, -0.2)
    import time as real_time
    real_time.sleep(0.05)
    ros.rclpy.run_for(0.1)
    assert heard and heard[-1] == (0.12, -0.2)
    ros.rclpy.run_for(1.0)                                      # planner silent for a second
    assert heard[-1] == (0.0, 0.0)                              # watchdog: zero
    n = len(heard)
    ros.rclpy.run_for(3.0)
    assert len(heard) - n < 25                                  # then quiet, so a gamepad can drive
    relay.sock.close()
    bridge.close()


def synth_frame(env, tags):
    from swarm_cps.core.synth import Scene
    sc = Scene(env.fm, seed=3)
    for t in tags:
        sc.add_tag(*t[:4], z=t[4], size=0.15)
    return sc.render(ss=1)


def test_perception_node_turns_frames_into_stamped_poses(ros, tmp_path):
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    snap = str(tmp_path / 'cam.jpg')
    ros.Node.OVERRIDES['swarm_perception'] = dict(source='ros_compressed', snapshot=snap)
    ros.rclpy.init()
    node = ros.mods['perception_node'].Perception()
    listener = ros.Node('listener')
    poses, vels = [], []
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                   history=HistoryPolicy.KEEP_LAST, depth=1)
    listener.create_subscription(PoseStamped, '/rover2/gps/pose', poses.append, q)
    listener.create_subscription(TwistStamped, '/rover2/gps/vel', vels.append, q)
    cam = listener.create_publisher(CompressedImage, '/image_raw/compressed', qos_profile_sensor_data)
    for k in range(6):                                          # rover2 driving east at 0.15 m/s, seen at 10 Hz
        x = 1.5 + 0.015 * k
        img = synth_frame(ros, [(2, x, 1.2, 0.0, 0.12), (1, 0.8, 0.9, 1.0, 0.12)])
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        m = CompressedImage()
        m.format = 'jpeg'
        m.data = jpg.tobytes()
        m.header.stamp.sec = 1700001000
        m.header.stamp.nanosec = int(k * 1e8)
        cam.publish(m)
    assert len(poses) == 6 and len(vels) == 6
    p = poses[-1]
    assert p.header.frame_id == 'arena'
    assert (p.header.stamp.sec, p.header.stamp.nanosec) == (1700001000, 500000000)
    assert abs(p.pose.position.x - 1.575) < 0.02 and abs(p.pose.position.y - 1.2) < 0.02
    yaw = 2 * math.atan2(p.pose.orientation.z, p.pose.orientation.w)
    assert abs(yaw) < math.radians(4)
    assert 0.05 < vels[-1].twist.linear.x < 0.25
    ros.rclpy.run_for(1.1)                                      # the health tick
    h = json.loads(node.health_pub.sent[-1].data)
    assert h['calibrated'] and h['rate_hz'] > 0 and h['seen_hz']['rover2'] > 0
    assert os.path.getsize(snap) > 1000
    assert any('rover3' in msg for lvl, msg in node.get_logger().records if lvl == 'warn')   # it says who is missing


def test_perception_node_without_a_floor_calibration_says_so_and_publishes_nothing(ros):
    os.remove(os.path.join(ros.cfg, 'camera_pose.yaml'))
    ros.Node.OVERRIDES['swarm_perception'] = dict(source='ros_compressed', snapshot='')
    ros.rclpy.init()
    node = ros.mods['perception_node'].Perception()
    assert any('NO FLOOR CALIBRATION' in msg for lvl, msg in node.get_logger().records if lvl == 'error')
    node.process(np.zeros((1080, 1920), np.uint8), 0.0, 1.7e9)
    ros.rclpy.run_for(1.1)
    assert all(not p.sent for p in node.pose_pub.values())


class FakeCamera:
    """cv2.VideoCapture stand-in: 1080p frames at ~100 fps, with one gap and
    one burst of failed reads."""
    def __init__(self, *a):
        self.n = 0
        self.props = {}

    def isOpened(self):
        return True

    def set(self, k, v):
        self.props[k] = v
        return True

    def get(self, k):
        if k == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*'MJPG'))
        return float(self.props.get(k, 0))

    def grab(self):
        import time
        self.n += 1
        time.sleep(0.05 if self.n == 60 else 0.01)
        return not (80 <= self.n < 83)

    def retrieve(self):
        img = np.full((1080, 1920, 3), self.n % 250, np.uint8)
        return True, img

    def release(self):
        pass


def test_capture_thread_hands_out_the_newest_frame_and_counts_trouble(monkeypatch):
    import time
    from swarm_cps import capture as C
    monkeypatch.setattr(C.cv2, 'VideoCapture', FakeCamera)
    monkeypatch.setattr(C, 'find_device', lambda hint='See3CAM': '/dev/video9')
    cap = C.Capture('auto', 1920, 1080, 100, 'MJPG', controls=None)
    info = cap.open()
    assert info['fourcc'] == 'MJPG' and info['device'] == '/dev/video9'
    cap.start()
    seq, seen = 0, []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.3:
        got = cap.wait_new(seq, 0.5)
        assert got is not None
        seq, t, gray = got
        assert gray.shape == (1080, 1920) and gray.dtype == np.uint8          # converted to grey
        seen.append(seq)
        time.sleep(0.03)                                        # a slow consumer
    cap.stop()
    assert seen == sorted(seen) and len(set(seen)) == len(seen)
    assert seen[-1] - seen[0] > 1.5 * len(seen)                 # it skipped stale frames instead of queueing them
    assert cap.n_late >= 1 and cap.n_fail == 3                  # the gap and the failed reads were counted
    assert 60 < cap.measured_fps() < 110


def test_capture_refuses_a_resolution_the_calibration_does_not_match(monkeypatch):
    from swarm_cps import capture as C

    class Small(FakeCamera):
        def get(self, k):
            if k == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280.0
            if k == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720.0
            return FakeCamera.get(self, k)
    monkeypatch.setattr(C.cv2, 'VideoCapture', Small)
    with pytest.raises(RuntimeError, match='1280x720'):
        C.Capture('/dev/video0', 1920, 1080, 60).open()


def test_node_entry_points_start_and_shut_down_cleanly(ros):
    ros.Node.OVERRIDES['swarm_planner'].update(decider='schedule', robot_output=True)
    ros.Node.OVERRIDES['swarm_foreman'].update(backend='mock')
    for name in ('planner_node', 'sim_node', 'foreman_node'):
        ros.mods[name].main()
        assert not ros.rclpy.ok()


def test_doctor_reports_instead_of_crashing_when_nothing_is_running(ros, capsys):
    rc = ros.mods['doctor'].main()
    out = capsys.readouterr().out
    assert rc == 1
    assert 'no poses on /rover1/gps/pose' in out and 'RESULT: not ready' in out
    assert '[FAIL] rover1: 0 subscriber(s) on /cmd_vel in domain 1' in out


def test_console_commands(ros, monkeypatch, capsys):
    import io
    from std_msgs.msg import String
    ros.rclpy.init()
    spy = ros.Node('spy')
    seen = []
    spy.create_subscription(String, '/swarm/control', lambda m: seen.append(('control', m.data)), 10)
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    spy.create_subscription(String, '/swarm/instruction', lambda m: seen.append(('say', m.data)), QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST, depth=1))
    ros.rclpy.shutdown()
    ros.rclpy._default._ok = False
    monkeypatch.setattr(sys, 'stdin', io.StringIO('go\n\nsay %s\ndecider rule\nnonsense\nquit\n' % CATTLE))
    spy._domain = 0
    ros.mods['console'].main()
    assert seen == [('control', 'arm'), ('control', 'stop'), ('say', CATTLE), ('control', 'decider rule'),
                    ('control', 'stop')]
    assert 'not a command' in capsys.readouterr().out
