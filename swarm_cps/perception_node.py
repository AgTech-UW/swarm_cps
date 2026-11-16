"""Faux-GPS: camera in, rover poses out.

Publishes, per rover, exactly the Faux-GPS paper's pose-source contract:

  /<name>/gps/pose   geometry_msgs/PoseStamped   arena frame, metres, stamped with the frame's acquisition time
  /<name>/gps/vel    geometry_msgs/TwistStamped  linear.x = forward speed along the heading, angular.z = yaw rate

both best-effort, volatile, depth 1, and only when a new frame produced a new
detection (nothing is extrapolated). So this node is a drop-in replacement for
faux_gps_bridge.py in front of any planner written to that contract.

  /swarm/perception  std_msgs/String (JSON), 1 Hz: frame rate, late frames, detection time, who is seen

Where the frames come from (parameter `source`):

  v4l2            (default) the camera is opened here; no image crosses DDS
  ros_compressed  frames from an existing usb_cam on <image_topic>/compressed,
                  for when the old launch file is the only thing that will open the camera
  apriltag_ros    tag corners from an existing apriltag_ros on /detections

    ros2 run swarm_cps perception
    ros2 run swarm_cps perception --ros-args -p show:=true
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String

from .capture import Capture
from .core.localize import Localizer
from .core.tags import Detector, TagDetection, corners_from_apriltag_msg
from .core.viz import draw_floor_grid
from .settings import Settings

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None

POSE_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


class Perception(Node):
    def __init__(self):
        super().__init__('swarm_perception')
        self.declare_parameter('config_dir', '')
        self.declare_parameter('source', 'v4l2')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('show', False)
        self.declare_parameter('snapshot', '/tmp/swarm_camera.jpg')
        self.declare_parameter('legacy_gps_robot', '')
        log = self.get_logger()
        self.st = Settings(self.get_parameter('config_dir').value or None)
        self.source = self.get_parameter('source').value
        self.show = bool(self.get_parameter('show').value)
        self.snapshot = self.get_parameter('snapshot').value
        cc = self.st.camera.get('capture') or {}
        dc = self.st.camera.get('detector') or {}
        self.latency = float(cc.get('latency', 0.02))
        self.fps = float(cc.get('fps', 30))
        self.cam = self.st.camera_model()
        self.fm = self.st.floor_map(self.cam)
        self.loc = None
        self.detector = None
        if self.source != 'apriltag_ros':
            self.detector = Detector(dc.get('backend', 'auto'), decimate=float(dc.get('decimate', 1.0)),
                                     threads=int(dc.get('threads', 4)), min_margin=float(dc.get('min_margin', 25)))
            log.info('tag detector: %s, decimate %.1f' % (self.detector.backend, self.detector.decimate))
            if self.detector.backend == 'aruco':
                log.warn('using the OpenCV detector. It misses far, slanted tags that the AprilTag library finds: '
                         'pip install --no-deps pupil-apriltags')
        if self.fm is None:
            log.error('NO FLOOR CALIBRATION (%s is missing). Poses cannot be published. Run: '
                      'ros2 run swarm_cps calibrate floor' % self.st.pose_path)
        else:
            anchors = {i: (a['x'], a['y'], a.get('z', 0.0)) for i, a in self.st.anchors().items()}
            self.loc = Localizer(self.detector, self.fm, self.st.robot_tags(), anchors=anchors,
                                 rescan_period=float(dc.get('rescan_period', 5.0)), arena_roi=dc.get('roi'))
            m = self.fm.describe()
            log.info('camera at (%.2f, %.2f, %.2f) m, looking %.0f deg below horizontal'
                     % (m['position'][0], m['position'][1], m['position'][2], m['pitch_down_deg']))
        self.alarm = float(self.st.camera.get('anchor_alarm_m', 0.03))
        self.pose_pub = {n: self.create_publisher(PoseStamped, '/%s/gps/pose' % n, POSE_QOS) for n in self.st.names}
        self.vel_pub = {n: self.create_publisher(TwistStamped, '/%s/gps/vel' % n, POSE_QOS) for n in self.st.names}
        self.health_pub = self.create_publisher(String, '/swarm/perception', 10)
        self.legacy = None
        legacy = self.get_parameter('legacy_gps_robot').value
        if legacy:
            try:                                              # the course's /tractor/gps, for the labs
                from turtlesim.msg import Pose as TurtlePose
                self._TurtlePose = TurtlePose
                self.legacy = (legacy, self.create_publisher(TurtlePose, '/tractor/gps', 10))
            except Exception as e:
                log.warn('legacy /tractor/gps not available (%s)' % e)
        # health counters
        self.seq = 0
        self.n_proc = 0
        self.n_skipped = 0
        self.t_detect = []
        self.seen = {n: 0 for n in self.st.names}
        self.last_seen = {n: None for n in self.st.names}
        self.t_health = time.monotonic()
        self.t_snap = 0.0
        self.last_frame = None
        self.last_out = {}
        self.cap = None
        if self.source == 'v4l2':
            self.cap = Capture(cc.get('device', 'auto'), cc.get('width', 1920), cc.get('height', 1080), self.fps,
                               cc.get('fourcc', 'MJPG'), cc.get('controls'), log=lambda s: log.warn(str(s)))
            info = self.cap.open()
            log.info('camera %(device)s: %(width)dx%(height)d %(fourcc)s, driver says %(fps).0f fps' % info)
            if info['fourcc'] != cc.get('fourcc', 'MJPG'):
                log.warn('camera is sending %s, not %s: expect a lower frame rate' % (info['fourcc'], cc.get('fourcc', 'MJPG')))
            self.cap.start()
            self.create_timer(0.003, self.poll_camera)
        elif self.source == 'ros_compressed':
            from sensor_msgs.msg import CompressedImage
            topic = self.get_parameter('image_topic').value.rstrip('/') + '/compressed'
            self.create_subscription(CompressedImage, topic, self.on_compressed, qos_profile_sensor_data)
            log.info('frames from %s' % topic)
        elif self.source == 'apriltag_ros':
            from apriltag_msgs.msg import AprilTagDetectionArray
            self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, qos_profile_sensor_data)
            log.info('tag corners from /detections')
        else:
            raise RuntimeError('unknown source %r' % self.source)
        self.create_timer(1.0, self.health)

    # ---- sources ----
    def poll_camera(self):
        got = self.cap.poll(self.seq)
        if got is None:
            return
        seq, t_mono, gray = got
        if self.seq and seq - self.seq > 1:
            self.n_skipped += seq - self.seq - 1               # frames that arrived while we were busy
        self.seq = seq
        now_mono = time.monotonic()
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        stamp = now_ros - (now_mono - t_mono) - self.latency
        self.process(gray, t_mono, stamp)

    def on_compressed(self, msg):
        gray = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        self.process(gray, stamp, stamp)

    def on_detections(self, msg):
        if self.loc is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        found = {}
        for d in msg.detections:
            if getattr(d, 'hamming', 0) != 0:
                continue
            found[int(d.id)] = TagDetection(int(d.id), corners_from_apriltag_msg(d.corners),
                                            margin=float(getattr(d, 'decision_margin', 100.0)))
        self.n_proc += 1
        self.publish(self.loc.process_detections(found, stamp), stamp)

    # ---- one frame ----
    def process(self, gray, t, stamp):
        self.last_frame = gray
        if self.loc is None:
            self.n_proc += 1
            return
        t0 = time.monotonic()
        out = self.loc.process(gray, t)
        self.t_detect.append(time.monotonic() - t0)
        self.n_proc += 1
        self.publish(out, stamp)

    def publish(self, out, stamp):
        sec = int(math.floor(stamp))
        nsec = int((stamp - sec) * 1e9)
        self.last_out = out
        for name, o in out.items():
            ps = PoseStamped()
            ps.header.frame_id = 'arena'
            ps.header.stamp.sec, ps.header.stamp.nanosec = sec, nsec
            ps.pose.position.x, ps.pose.position.y = o['x'], o['y']
            ps.pose.orientation.z = math.sin(o['yaw'] / 2)
            ps.pose.orientation.w = math.cos(o['yaw'] / 2)
            self.pose_pub[name].publish(ps)
            tw = TwistStamped()
            tw.header.frame_id = 'arena'
            tw.header.stamp.sec, tw.header.stamp.nanosec = sec, nsec
            tw.twist.linear.x = o['v']
            tw.twist.angular.z = o['w']
            self.vel_pub[name].publish(tw)
            self.seen[name] += 1
            self.last_seen[name] = time.monotonic()
            if self.legacy is not None and self.legacy[0] == name:
                m = self._TurtlePose()
                m.x, m.y, m.theta = float(o['x']), float(o['y']), float(o['yaw'])
                m.linear_velocity, m.angular_velocity = float(o['v']), float(o['w'])
                self.legacy[1].publish(m)

    # ---- once a second ----
    def health(self):
        log = self.get_logger()
        now = time.monotonic()
        dt = max(now - self.t_health, 1e-3)
        self.t_health = now
        rate = self.n_proc / dt
        td = self.t_detect
        h = dict(rate_hz=round(rate, 1), detect_ms=round(1e3 * float(np.mean(td)), 1) if td else None,
                 detect_max_ms=round(1e3 * float(np.max(td)), 1) if td else None, skipped=self.n_skipped,
                 seen_hz={n: round(c / dt, 1) for n, c in self.seen.items()}, calibrated=self.loc is not None,
                 source=self.source)
        if self.cap is not None:
            h.update(camera_fps=round(self.cap.measured_fps(), 1), late_frames=self.cap.n_late,
                     max_gap_ms=round(1e3 * self.cap.max_gap, 1), camera_error=self.cap.error)
            if self.cap.error:
                log.error('camera: %s' % self.cap.error)
            elif self.cap.n_frames > 3 * self.fps and self.cap.measured_fps() < 0.8 * self.fps:
                log.warn('camera delivers %.1f fps, asked for %.0f. Check exposure (must be shorter than a frame), '
                         'the USB port (lsusb -t must show 5000M), and the cable.' % (self.cap.measured_fps(), self.fps),
                         throttle_duration_sec=10.0)
        if self.loc is not None:
            h.update(anchor_error_m=self.loc.anchor_error_m, anchors_seen=self.loc.anchor_seen,
                     full_scans=self.loc.stats['full_scans'])
            if self.loc.anchor_error_m is not None and self.loc.anchor_error_m > self.alarm:
                log.warn('floor reference tags appear %.0f mm from where they were calibrated: the camera has '
                         'moved. Recalibrate (ros2 run swarm_cps calibrate floor).' % (1e3 * self.loc.anchor_error_m),
                         throttle_duration_sec=10.0)
            missing = [n for n, t in self.last_seen.items() if t is None or now - t > 1.0]
            if missing:
                log.warn('not seeing: %s' % ', '.join(missing), throttle_duration_sec=5.0)
        msg = String()
        msg.data = json.dumps(h)
        self.health_pub.publish(msg)
        self.n_proc = 0
        self.t_detect = []
        self.seen = {n: 0 for n in self.st.names}
        if self.last_frame is not None and cv2 is not None and (self.show or self.snapshot):
            self.draw_debug(h)

    def draw_debug(self, h):
        img = cv2.cvtColor(self.last_frame, cv2.COLOR_GRAY2BGR)
        for name, o in self.last_out.items():
            p = (int(o['px'][0]), int(o['px'][1]))
            cv2.circle(img, p, 14, (0, 220, 0), 2)
            cv2.putText(img, '%s (%.2f, %.2f) %d deg' % (name, o['x'], o['y'], round(math.degrees(o['yaw']))),
                        (p[0] + 18, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
        if self.fm is not None:
            ar = self.st.arena()
            draw_floor_grid(img, self.fm, (ar.x0, ar.x1, ar.y0, ar.y1))
        cv2.putText(img, '%.0f Hz  detect %s ms' % (h['rate_hz'], h['detect_ms']), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if self.snapshot:
            try:
                cv2.imwrite(self.snapshot, small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            except Exception:
                pass
        if self.show:
            try:
                cv2.imshow('swarm camera', small)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn('cannot open a window (%s); see %s instead' % (e, self.snapshot))
                self.show = False

    def close(self):
        if self.cap is not None:
            self.cap.stop()


def main(args=None):
    rclpy.init(args=args)
    node = Perception()
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
