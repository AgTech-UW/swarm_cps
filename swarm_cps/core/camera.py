"""Camera intrinsics and the floor map for an obliquely mounted camera.

Why this exists
---------------
With a ceiling camera looking straight down, taking the tag pose straight from
the AprilTag pose estimator works well enough. With the camera in a corner,
looking across the floor, it does not: the estimator recovers range from the
apparent SIZE of a small, foreshortened tag, so range noise grows with the
square of distance, and a flat tag seen at a shallow angle has two nearly
equivalent orientations that the estimator flips between.

So we do not use the tag's estimated 3-D pose at all. We know something the
estimator does not: every rover tag sits on a horizontal plane at a known
height. A pixel plus a known plane is a unique 3-D point. We therefore

  1. calibrate where the camera is, once, from markers at known floor positions
     (FloorMap.solve), and
  2. for every detected tag, undistort its four corner pixels, cast the four
     rays, and intersect them with the plane z = tag height.

Position then depends on pixel accuracy times the ground-sample distance (a few
millimetres) and heading comes from the tag's four corners on the floor plane,
which cannot flip.

Frames
------
world/arena: x, y on the floor in metres, z up. Camera: OpenCV convention
(x right, y down, z forward).  X_cam = R @ X_world + t.
"""
import math

import numpy as np

try:                                    # OpenCV is only needed for the PnP refinement
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------
#  Intrinsics
# --------------------------------------------------------------------------

class CameraModel:
    """Pinhole + OpenCV distortion (plumb_bob 5 coefficients or rational 8).

    Distortion and its inverse are implemented here in numpy rather than left
    to cv2.undistortPoints, whose default five fixed-point iterations are not
    enough for a lens this wide (k1 = -0.72 in the lab calibration) near the
    image edge. undistort() iterates to convergence and reports the residual.
    """

    def __init__(self, K, D, size, name='camera'):
        self.K = np.asarray(K, float).reshape(3, 3)
        d = np.zeros(8)
        dd = np.asarray(D, float).ravel()
        d[:min(8, dd.size)] = dd[:8]
        self.D = d
        self.size = (int(size[0]), int(size[1]))          # (width, height)
        self.name = name

    # ---- loading ----
    @classmethod
    def from_ros_yaml(cls, path):
        """Read a camera_calibration / camera_info_manager YAML (the lab's
        default_cam.yaml)."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        K = d['camera_matrix']['data']
        D = d['distortion_coefficients']['data']
        return cls(K, D, (d['image_width'], d['image_height']), d.get('camera_name', 'camera'))

    def scaled(self, width, height):
        """Intrinsics for the same sensor area read out at another resolution
        (pure scaling). Only valid if the new mode shows the same field of view;
        a cropped mode needs its own calibration."""
        sx, sy = width / self.size[0], height / self.size[1]
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return CameraModel(K, self.D, (width, height), self.name)

    # ---- distortion ----
    def _distort_norm(self, xn):
        """Ideal normalised coords (N, 2) -> distorted normalised coords."""
        k1, k2, p1, p2, k3, k4, k5, k6 = self.D
        x, y = xn[:, 0], xn[:, 1]
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        rad = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], 1)

    def normalized_to_pixel(self, xn):
        """Ideal normalised coords (N, 2) -> pixels (N, 2), with distortion."""
        xd = self._distort_norm(np.atleast_2d(np.asarray(xn, float)))
        return np.stack([self.K[0, 0] * xd[:, 0] + self.K[0, 2],
                         self.K[1, 1] * xd[:, 1] + self.K[1, 2]], 1)

    def undistort(self, px, iters=60, tol=1e-10):
        """Pixels (N, 2) -> ideal normalised coords (N, 2).

        Damped fixed-point iteration followed by Newton steps on the true
        forward model, so it converges for strong barrel distortion too."""
        px = np.atleast_2d(np.asarray(px, float))
        yd = np.stack([(px[:, 0] - self.K[0, 2]) / self.K[0, 0],
                       (px[:, 1] - self.K[1, 2]) / self.K[1, 1]], 1)
        x = yd.copy()
        k1, k2, p1, p2, k3, k4, k5, k6 = self.D
        for _ in range(15):                                  # fixed point: good start
            r2 = (x * x).sum(1)
            r4 = r2 * r2
            r6 = r4 * r2
            rad = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
            rad = np.where(np.abs(rad) < 1e-3, 1e-3, rad)
            dx = 2 * p1 * x[:, 0] * x[:, 1] + p2 * (r2 + 2 * x[:, 0] ** 2)
            dy = p1 * (r2 + 2 * x[:, 1] ** 2) + 2 * p2 * x[:, 0] * x[:, 1]
            x = np.stack([(yd[:, 0] - dx) / rad, (yd[:, 1] - dy) / rad], 1)
        h = 1e-6
        for _ in range(iters):                               # Newton polish
            f = self._distort_norm(x) - yd
            if np.abs(f).max() < tol:
                break
            fx = (self._distort_norm(x + [h, 0]) - yd - f) / h
            fy = (self._distort_norm(x + [0, h]) - yd - f) / h
            a, b, c, d = fx[:, 0], fy[:, 0], fx[:, 1], fy[:, 1]
            det = a * d - b * c
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            step = np.stack([(d * f[:, 0] - b * f[:, 1]) / det,
                             (-c * f[:, 0] + a * f[:, 1]) / det], 1)
            step = np.clip(step, -0.5, 0.5)
            x = x - step
        return x

    def undistort_residual_px(self, px):
        """How well undistort() inverted the model, in pixels (max over points).
        Anything above ~0.05 px means the pixel is outside the region the
        calibration model can describe."""
        px = np.atleast_2d(np.asarray(px, float))
        back = self.normalized_to_pixel(self.undistort(px))
        return float(np.abs(back - px).max())

    def fov_deg(self):
        """True horizontal and vertical field of view through the lens centre,
        distortion included (the pinhole K alone understates a wide lens)."""
        w, h = self.size
        cx, cy = self.K[0, 2], self.K[1, 2]
        n = self.undistort([[0, cy], [w - 1, cy], [cx, 0], [cx, h - 1]])
        hf = math.degrees(math.atan(-n[0, 0]) + math.atan(n[1, 0]))
        vf = math.degrees(math.atan(-n[2, 1]) + math.atan(n[3, 1]))
        return hf, vf


# --------------------------------------------------------------------------
#  Extrinsics: where the camera is relative to the floor
# --------------------------------------------------------------------------

def _rot_from_rvec(rvec):
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rvec, float).ravel() / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _rvec_from_rot(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if abs(th - math.pi) < 1e-6:                              # near 180 deg: use the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[i] / max(k[i], 1e-12)
        return th * k / max(np.linalg.norm(k), 1e-12)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * v / (2 * math.sin(th))


class FloorMap:
    """Camera pose relative to the arena, and the pixel <-> floor-plane maps."""

    def __init__(self, camera, R, t):
        self.camera = camera
        self.R = np.asarray(R, float).reshape(3, 3)
        self.t = np.asarray(t, float).reshape(3)
        self.center = -self.R.T @ self.t                       # camera position in the arena
        self.info = {}

    # ---- construction ----
    @classmethod
    def from_mount(cls, camera, position, look_at, roll_deg=0.0):
        """A camera at `position` (x, y, z) aimed at the point `look_at`
        (x, y, z), image 'up' as close to world up as possible. Used by the
        mount planner and the synthetic tests."""
        C = np.asarray(position, float)
        f = np.asarray(look_at, float) - C
        f = f / np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        if np.linalg.norm(r) < 1e-9:                           # looking straight down
            r = np.array([1.0, 0.0, 0.0])
        r = r / np.linalg.norm(r)
        d = np.cross(f, r)                                     # image down
        a = math.radians(roll_deg)
        r2 = math.cos(a) * r + math.sin(a) * d
        d2 = -math.sin(a) * r + math.cos(a) * d
        R = np.stack([r2, d2, f], 0)
        return cls(camera, R, -R @ C)

    @classmethod
    def solve(cls, camera, pixels, world, refine=True):
        """Camera pose from N >= 4 points with known arena coordinates.

        pixels (N, 2) are raw (distorted) image points, world (N, 3) their
        arena positions in metres. Points are undistorted first, so the pose
        problem is solved in ideal normalised coordinates.

        Coplanar points (the usual case: markers on the floor) are initialised
        from the plane homography, which has a unique physical solution for a
        camera above the floor; the result is then refined by Levenberg-
        Marquardt on reprojection error. Returns a FloorMap whose .info holds
        the residuals."""
        px = np.asarray(pixels, float).reshape(-1, 2)
        W = np.asarray(world, float).reshape(-1, 3)
        n = len(px)
        if n < 4:
            raise ValueError('need at least 4 reference points, got %d' % n)
        xn = camera.undistort(px)
        planar = float(np.ptp(W[:, 2])) < 1e-3
        if planar:
            R, t = _pose_from_plane(xn, W)
        else:
            if cv2 is None or n < 6:
                raise ValueError('non-coplanar reference points need OpenCV and at least 6 points')
            ok, rvec, tvec = cv2.solvePnP(W.reshape(-1, 1, 3), xn.reshape(-1, 1, 2), np.eye(3), None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise ValueError('solvePnP failed')
            R, t = _rot_from_rvec(rvec.ravel()), tvec.ravel()
        if refine and cv2 is not None and n >= 4:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    W.reshape(-1, 1, 3).astype(np.float64), xn.reshape(-1, 1, 2).astype(np.float64),
                    np.eye(3), None, rvec=_rvec_from_rot(R).reshape(3, 1).copy(),
                    tvec=np.asarray(t, float).reshape(3, 1).copy(),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    R2, t2 = _rot_from_rvec(rvec.ravel()), tvec.ravel()
                    if _reproj_rms(xn, W, R2, t2) <= _reproj_rms(xn, W, R, t) + 1e-12:
                        R, t = R2, t2
            except Exception:
                pass                                           # keep the closed-form pose
        fm = cls(camera, R, t)
        fm.info = fm.residuals(px, W)
        fm.info['n_points'] = n
        fm.info['planar'] = planar
        return fm

    # ---- maps ----
    def pixel_to_plane(self, px, z=0.0):
        """Raw pixels (N, 2) -> arena (x, y) (N, 2) on the horizontal plane at
        height z. Points whose ray does not hit the plane in front of the
        camera come back as NaN."""
        xn = self.camera.undistort(px)
        d = np.concatenate([xn, np.ones((len(xn), 1))], 1) @ self.R      # rows are R.T @ ray
        dz = d[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (z - self.center[2]) / dz
        P = self.center[None, :] + s[:, None] * d
        bad = ~np.isfinite(s) | (s <= 0)
        P[bad] = np.nan
        return P[:, :2]

    def world_to_pixel(self, W):
        """Arena points (N, 3) -> raw pixels (N, 2) and a mask of points in
        front of the camera."""
        W = np.atleast_2d(np.asarray(W, float))
        Xc = W @ self.R.T + self.t
        front = Xc[:, 2] > 1e-6
        z = np.where(front, Xc[:, 2], 1.0)
        px = self.camera.normalized_to_pixel(np.stack([Xc[:, 0] / z, Xc[:, 1] / z], 1))
        return px, front

    def in_image(self, px, margin=0.0):
        w, h = self.camera.size
        px = np.atleast_2d(px)
        return (px[:, 0] >= margin) & (px[:, 0] <= w - 1 - margin) & \
               (px[:, 1] >= margin) & (px[:, 1] <= h - 1 - margin)

    def ground_sample(self, xy, z=0.0):
        """How far a 1-pixel error moves a point on the plane at height z:
        returns (metres per pixel across the line of sight, metres per pixel
        along it) at arena point xy. This is the conversion from corner noise
        to position noise, and the number the mount planner maximises."""
        xy = np.asarray(xy, float).reshape(2)
        px, front = self.world_to_pixel([[xy[0], xy[1], z]])
        if not front[0]:
            return float('nan'), float('nan')
        a = self.pixel_to_plane(px + [[1.0, 0.0]], z)[0] - xy
        b = self.pixel_to_plane(px + [[0.0, 1.0]], z)[0] - xy
        J = np.stack([a, b], 1)                                # metres per pixel, 2x2
        sv = np.linalg.svd(J, compute_uv=False)
        return float(sv[1]), float(sv[0])

    def residuals(self, pixels, world):
        """Fit quality on a set of reference points."""
        px = np.asarray(pixels, float).reshape(-1, 2)
        W = np.asarray(world, float).reshape(-1, 3)
        back, _ = self.world_to_pixel(W)
        e_px = np.linalg.norm(back - px, axis=1)
        e_m = np.zeros(len(W))
        for i in range(len(W)):
            p = self.pixel_to_plane(px[i:i + 1], W[i, 2])[0]
            e_m[i] = np.linalg.norm(p - W[i, :2]) if np.all(np.isfinite(p)) else np.nan
        return dict(reproj_rms_px=float(np.sqrt(np.mean(e_px ** 2))), reproj_max_px=float(e_px.max()),
                    floor_rms_m=float(np.sqrt(np.nanmean(e_m ** 2))), floor_max_m=float(np.nanmax(e_m)),
                    per_point_m=[float(v) for v in e_m])

    def describe(self):
        """Human-readable mount description (for logs and the calibration tool)."""
        f = self.R[2]                                          # optical axis in the arena
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -f[2]))))   # + = looking down
        yaw = math.degrees(math.atan2(f[1], f[0]))
        return dict(position=[float(v) for v in self.center], pitch_down_deg=pitch, yaw_deg=yaw)

    # ---- storage ----
    def to_dict(self):
        return dict(R=[float(v) for v in self.R.ravel()], t=[float(v) for v in self.t],
                    image_size=list(self.camera.size), info=_plain(self.info), mount=self.describe())

    @classmethod
    def from_dict(cls, camera, d):
        fm = cls(camera, np.array(d['R'], float).reshape(3, 3), np.array(d['t'], float))
        fm.info = d.get('info', {})
        return fm


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _reproj_rms(xn, W, R, t):
    Xc = W @ R.T + t
    if np.any(Xc[:, 2] <= 1e-9):
        return float('inf')
    e = Xc[:, :2] / Xc[:, 2:3] - xn
    return float(np.sqrt(np.mean(np.sum(e * e, 1))))


def _homography_dlt(src, dst):
    """Least-squares homography dst ~ H src (both (N, 2)), Hartley-normalised."""
    def norm(p):
        c = p.mean(0)
        s = math.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return (p - c) * s, T
    a, Ta = norm(np.asarray(src, float))
    b, Tb = norm(np.asarray(dst, float))
    A = []
    for (x, y), (u, v) in zip(a, b):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ H @ Ta
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def _pose_from_plane(xn, W):
    """Pose from coplanar points (all at the same height z0): normalised image
    coords = H @ [X, Y, 1], with H = [r1 r2 (r3 z0 + t)] up to scale."""
    z0 = float(W[:, 2].mean())
    H = _homography_dlt(W[:, :2], xn)
    best = None
    for sign in (1.0, -1.0):
        Hs = sign * H
        lam = 2.0 / (np.linalg.norm(Hs[:, 0]) + np.linalg.norm(Hs[:, 1]))
        r1, r2, tt = lam * Hs[:, 0], lam * Hs[:, 1], lam * Hs[:, 2]
        r3 = np.cross(r1, r2)
        U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], 1))    # nearest true rotation
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        t = tt - R[:, 2] * z0
        Xc = W @ R.T + t
        if np.all(Xc[:, 2] > 0):                               # points in front of the camera
            err = _reproj_rms(xn, W, R, t)
            if best is None or err < best[0]:
                best = (err, R, t)
    if best is None:
        raise ValueError('no camera pose puts the reference points in front of the camera; '
                         'check the point order and the arena coordinates')
    return best[1], best[2]


def trilaterate(dist_a, dist_b, a, b, side=1.0):
    """Arena position of a point from its tape-measured distances to two known
    floor points a and b. Two tapes from two marks is the easy accurate way to
    survey a floor (squaring a tape against a wall by eye is not). `side` picks
    which side of the line a->b the point is on: +1 = left of a->b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.linalg.norm(b - a))
    if d < 1e-9:
        raise ValueError('reference marks coincide')
    x = (dist_a ** 2 - dist_b ** 2 + d ** 2) / (2 * d)
    h2 = dist_a ** 2 - x ** 2
    if h2 < -1e-3 * max(1.0, dist_a ** 2):
        raise ValueError('distances %.3f and %.3f do not meet for marks %.3f m apart'
                         % (dist_a, dist_b, d))
    h = math.sqrt(max(h2, 0.0))
    u = (b - a) / d
    n = np.array([-u[1], u[0]])
    return a + x * u + side * h * n
