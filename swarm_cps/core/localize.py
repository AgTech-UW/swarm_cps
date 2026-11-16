"""Frame -> rover poses. This is the whole perception chain behind the "faux
GPS", kept free of ROS so it can be run on a saved image, a synthetic render
or a live camera with the same code.

    gray frame --Detector--> tag corners --FloorMap--> pose on the tag plane
               --per-robot mounting correction--> vehicle reference pose
               --PoseFilter--> pose + forward speed + yaw rate

Speed: a full-resolution search of a 1920x1080 frame is the slow part. Once a
tag has been found it is searched for only in a window around where it was
(a few milliseconds per rover), and the full frame is searched again only when
a rover is missing or every `rescan_period` seconds.
"""
import math

import numpy as np

from .tags import tag_floor_pose


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class RobotTag:
    """How one rover's tag is mounted.

    tag_height   height of the printed tag above the floor (m). Measure it; a
                 2 cm error is a 3-5 cm position error at the far end of an
                 oblique view.
    yaw_offset   rover forward direction minus the tag's "up" direction (rad).
                 Found by tools/swarm_calibrate.py yaw, so the tag can be
                 stuck on at any rotation.
    offset       position of the vehicle reference point (rear-axle centre)
                 relative to the tag centre, in the rover's own frame
                 (forward, left), metres.
    """
    def __init__(self, name, tag_id, tag_height=0.12, tag_size=0.15, yaw_offset=0.0, offset=(0.0, 0.0)):
        self.name = name
        self.tag_id = int(tag_id)
        self.tag_height = float(tag_height)
        self.tag_size = float(tag_size)
        self.yaw_offset = float(yaw_offset)
        self.offset = (float(offset[0]), float(offset[1]))

    def vehicle_pose(self, tag_pose):
        yaw = wrap(tag_pose['yaw'] + self.yaw_offset)
        c, s = math.cos(yaw), math.sin(yaw)
        x = tag_pose['x'] + self.offset[0] * c - self.offset[1] * s
        y = tag_pose['y'] + self.offset[0] * s + self.offset[1] * c
        return x, y, yaw


class PoseFilter:
    """Constant-velocity Kalman filter on [x, y, yaw, vx, vy, yaw_rate], the
    filter of the Faux-GPS paper (its Eq. 1-2): yaw innovations are wrapped,
    the time step is the interval between accepted measurements, forward speed
    is the velocity projected on the heading, and nothing is extrapolated
    between measurements. Added here: a gate that rejects a single wild
    measurement instead of letting it kick the velocity estimate, and a
    position sigma supplied per measurement, because from a corner camera a
    pixel is worth more millimetres far away than close up."""

    def __init__(self, accel_sigma=0.6, yaw_accel_sigma=3.0, yaw_sigma=math.radians(1.5), gate=5.0,
                 max_rejects=3, reset_after=1.0):
        self.qa, self.qw = accel_sigma, yaw_accel_sigma
        self.yaw_sigma = yaw_sigma
        self.gate = gate
        self.max_rejects = max_rejects
        self.reset_after = reset_after
        self.x = None
        self.P = None
        self.t = None
        self.rejects = 0
        self.n = 0

    def reset(self):
        self.x = self.P = self.t = None
        self.rejects = 0
        self.n = 0

    def update(self, t, x, y, yaw, pos_sigma=0.005):
        """Feed one measurement taken at time t (s). Returns True if it was
        accepted. Measurements that are not newer than the last one are
        dropped."""
        z = np.array([x, y, yaw], float)
        pos_sigma = max(float(pos_sigma), 0.001)
        if self.x is None or (t - self.t) > self.reset_after:
            self.x = np.array([x, y, yaw, 0.0, 0.0, 0.0])
            self.P = np.diag([pos_sigma ** 2, pos_sigma ** 2, self.yaw_sigma ** 2, 0.25, 0.25, 1.0])
            self.t = t
            self.rejects = 0
            self.n = 1
            return True
        dt = t - self.t
        if dt <= 1e-6:
            return False                                      # repeated or reordered stamp
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = np.zeros((6, 6))
        for (i, j, s) in ((0, 3, self.qa), (1, 4, self.qa), (2, 5, self.qw)):
            q[i, i] = s * s * dt ** 4 / 4
            q[i, j] = q[j, i] = s * s * dt ** 3 / 2
            q[j, j] = s * s * dt ** 2
        xp = F @ self.x
        Pp = F @ self.P @ F.T + q
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.diag([pos_sigma ** 2, pos_sigma ** 2, self.yaw_sigma ** 2])
        inn = z - H @ xp
        inn[2] = wrap(inn[2])
        S = H @ Pp @ H.T + R
        Sinv = np.linalg.inv(S)
        d2 = float(inn @ Sinv @ inn)
        if d2 > self.gate ** 2 and self.rejects < self.max_rejects and self.n > 3:
            self.rejects += 1                                 # one wild detection: ignore it
            return False
        if d2 > self.gate ** 2 and self.n > 3:
            # several in a row disagree with the filter: the filter is wrong
            self.x = np.array([x, y, yaw, 0.0, 0.0, 0.0])
            self.P = np.diag([pos_sigma ** 2, pos_sigma ** 2, self.yaw_sigma ** 2, 0.25, 0.25, 1.0])
            self.t = t
            self.rejects = 0
            self.n = 1
            return True
        K = Pp @ H.T @ Sinv
        self.x = xp + K @ inn
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(6) - K @ H) @ Pp
        self.t = t
        self.rejects = 0
        self.n += 1
        return True

    @property
    def pose(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    @property
    def forward_speed(self):
        return float(self.x[3] * math.cos(self.x[2]) + self.x[4] * math.sin(self.x[2]))

    @property
    def yaw_rate(self):
        return float(self.x[5])


class Localizer:
    """Holds the detector, the floor map, the fleet's tags and one filter per
    rover. process(gray, t) returns what was seen in this frame."""

    def __init__(self, detector, floor_map, robots, anchors=None, pixel_sigma=0.35,
                 rescan_period=5.0, lost_rescan=0.2, roi_scale=2.2, roi_min_px=70, arena_roi=None,
                 filter_kwargs=None):
        self.detector = detector
        self.fm = floor_map
        self.robots = {r.tag_id: r for r in robots}
        self.by_name = {r.name: r for r in robots}
        self.anchors = anchors or {}                # tag_id -> (x, y, z) of floor reference tags
        self.pixel_sigma = pixel_sigma
        self.rescan_period = rescan_period
        self.lost_rescan = lost_rescan
        self.roi_scale = roi_scale
        self.roi_min_px = roi_min_px
        self.arena_roi = arena_roi                  # (x0, y0, x1, y1) pixels, or None for the whole frame
        self.filters = {r.name: PoseFilter(**(filter_kwargs or {})) for r in robots}
        self._last_px = {}                          # tag_id -> (centre px, side px, time)
        self._last_full = -1e9
        self.anchor_error_m = None                  # latest check of the floor reference tags
        self.anchor_seen = 0
        self.stats = dict(frames=0, full_scans=0, roi_hits=0, roi_misses=0)

    # ---- detection scheduling ----
    def _roi(self, tag_id, t, shape):
        c, side, t0 = self._last_px[tag_id]
        age = max(0.0, t - t0)
        half = max(self.roi_min_px, self.roi_scale * side) * (1.0 + 2.0 * min(age, 0.5))
        x0, y0 = c[0] - half, c[1] - half
        x1, y1 = c[0] + half, c[1] + half
        return (max(0, int(x0)), max(0, int(y0)), min(shape[1], int(x1)), min(shape[0], int(y1)))

    def _detect(self, gray, t):
        """Windowed search around the tags being tracked. The whole frame is
        searched every `rescan_period` seconds, and at most every
        `lost_rescan` seconds while any rover is missing, so one rover out of
        view cannot drag the pose rate of the others down to the speed of a
        full search."""
        found = {}
        want = set(self.robots)
        tracked = [i for i in want if i in self._last_px and t - self._last_px[i][2] < 0.6]
        for i in tracked:
            for d in self.detector.detect(gray, self._roi(i, t, gray.shape)):
                if d.id == i:
                    found[i] = d
        missing = len(found) < len(want)
        if tracked:
            self.stats['roi_misses' if len(found) < len(tracked) else 'roi_hits'] += 1
        since = t - self._last_full
        full = since >= self.rescan_period or since < 0 or (missing and since >= self.lost_rescan)
        if full:
            for d in self.detector.detect(gray, self.arena_roi):
                found[d.id] = d
            self._last_full = t
            self.stats['full_scans'] += 1
        for i, d in found.items():
            self._last_px[i] = (d.center.copy(), d.side_px(), t)
        return found, full

    # ---- main entry ----
    def process(self, gray, t):
        """gray: uint8 frame; t: acquisition time in seconds (any monotonic
        clock). Returns dict name -> dict(x, y, yaw, v, w, raw=(x, y, yaw),
        sigma, px) for every rover seen in this frame and accepted by its
        filter."""
        self.stats['frames'] += 1
        found, full = self._detect(gray, t)
        return self.process_detections(found, t, check_anchors=full)

    def process_detections(self, found, t, check_anchors=True):
        """Same as process() but from detections made elsewhere (apriltag_ros).
        found: dict tag_id -> TagDetection."""
        out = {}
        for tag_id, det in found.items():
            r = self.robots.get(tag_id)
            if r is None:
                continue
            tp = tag_floor_pose(det, self.fm, r.tag_height, r.tag_size)
            if tp is None:
                continue
            x, y, yaw = r.vehicle_pose(tp)
            across, along = self.fm.ground_sample((tp['x'], tp['y']), r.tag_height)
            sigma = self.pixel_sigma * (along if math.isfinite(along) else 0.02)
            f = self.filters[r.name]
            if not f.update(t, x, y, yaw, sigma):
                continue
            fx, fy, fyaw = f.pose
            out[r.name] = dict(x=fx, y=fy, yaw=fyaw, v=f.forward_speed, w=f.yaw_rate, raw=(x, y, yaw),
                               sigma=sigma, px=(float(det.center[0]), float(det.center[1])),
                               size=tp['size'], t=t)
        if check_anchors and self.anchors:
            errs = []
            for tag_id, (ax, ay, az) in self.anchors.items():
                det = found.get(tag_id)
                if det is None:
                    continue
                p = self.fm.pixel_to_plane(det.corners, az).mean(0)
                if np.all(np.isfinite(p)):
                    errs.append(math.hypot(p[0] - ax, p[1] - ay))
            self.anchor_seen = len(errs)
            if errs:
                self.anchor_error_m = float(np.median(errs))
        return out
