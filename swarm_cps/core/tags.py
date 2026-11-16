"""AprilTag detection (two interchangeable backends) and tag -> floor pose.

Backends
--------
'apriltag'  the AprilTag 3 library through pupil_apriltags
            (pip install --no-deps pupil-apriltags). Best at small, slanted
            tags, which is what a corner-mounted camera sees.
'aruco'     OpenCV's ArUco detector with the AprilTag 36h11 dictionary. Ships
            with the OpenCV already in the ROS container, so it always works.
'auto'      'apriltag' if it imports, otherwise 'aruco'.

Whatever the backend, a detection's corners are returned in one fixed order
relative to the printed tag: top-left, top-right, bottom-right, bottom-left,
where "top" is the top of the tag image as published by the AprilTag project.
tests/test_perception.py renders tags with one library and detects them with
the other to check the two agree on id and on which edge is the top.
"""
import math

import numpy as np

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


def quad_center(c):
    """Image of the tag's centre: where the diagonals cross. Under perspective
    this, not the mean of the corners, is the projection of the true centre."""
    c = np.asarray(c, float).reshape(4, 2)
    p, r = c[0], c[2] - c[0]
    q, s = c[1], c[3] - c[1]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return c.mean(0)
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
    return p + t * r


_KEEP_ALIVE = []          # see Detector.__init__
APRILTAG_PIXEL_OFFSET = 0.5


class TagDetection:
    __slots__ = ('id', 'corners', 'center', 'margin')

    def __init__(self, tag_id, corners, margin=0.0):
        self.id = int(tag_id)
        self.corners = np.asarray(corners, float).reshape(4, 2)   # tl, tr, br, bl
        self.center = quad_center(self.corners)
        self.margin = float(margin)

    def side_px(self):
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))

    def min_side_px(self):
        c = self.corners
        return float(np.min([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def _aruco_dictionary():
    a = cv2.aruco
    if hasattr(a, 'getPredefinedDictionary'):
        return a.getPredefinedDictionary(a.DICT_APRILTAG_36h11)
    return a.Dictionary_get(a.DICT_APRILTAG_36h11)


def tag_image(tag_id, cells_px=20, border_cells=1):
    """Bitmap of AprilTag 36h11 `tag_id`: black square of 8x8 cells plus a white
    quiet zone of `border_cells` cells. Returns a uint8 image."""
    d = _aruco_dictionary()
    side = 8 * cells_px
    if hasattr(cv2.aruco, 'generateImageMarker'):
        img = cv2.aruco.generateImageMarker(d, int(tag_id), side)
    else:
        img = cv2.aruco.drawMarker(d, int(tag_id), side)
    img = np.ascontiguousarray(np.rot90(img, 2))      # OpenCV stores 36h11 rotated 180 deg from the official images
    b = border_cells * cells_px
    if b:
        img = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
    return img


class Detector:
    """detect(gray) -> [TagDetection]. `decimate` trades range for speed: 1.0
    looks at every pixel (needed for far tags from a corner mount), 2.0 is
    about three times faster and loses the tags that are only a few pixels
    per cell."""

    def __init__(self, backend='auto', decimate=1.0, threads=4, min_margin=25.0,
                 sharpening=0.25, sigma=0.0, min_side_px=12.0):
        self.decimate = float(decimate)
        self.min_margin = float(min_margin)
        self.min_side_px = float(min_side_px)
        self.backend = None
        self._at = None
        err = None
        if backend in ('auto', 'apriltag'):
            try:
                from pupil_apriltags import Detector as _AT
                # pupil_apriltags frees its tag family twice when a Detector is
                # garbage-collected, which corrupts the heap and aborts the
                # process some time later (reproduced: exit code 134 every run).
                # The detector lives as long as the process anyway, so never
                # run that destructor and keep every instance alive.
                _AT.__del__ = lambda self_: None
                self._at = _AT(families='tag36h11', nthreads=int(threads), quad_decimate=self.decimate,
                               quad_sigma=float(sigma), refine_edges=1, decode_sharpening=float(sharpening))
                _KEEP_ALIVE.append(self._at)
                self.backend = 'apriltag'
            except Exception as e:                            # not installed, or no wheel for this CPU
                err = e
                if backend == 'apriltag':
                    raise RuntimeError('pupil_apriltags is not usable (%s). Install it with '
                                       '"pip install --no-deps pupil-apriltags" or use backend aruco.' % e)
        if self.backend is None:
            if cv2 is None or not hasattr(cv2, 'aruco'):
                raise RuntimeError('no tag detector available: pupil_apriltags failed (%s) and this '
                                   'OpenCV has no aruco module' % err)
            self._setup_aruco()
            self.backend = 'aruco'

    # ---- OpenCV backend ----
    def _setup_aruco(self):
        a = cv2.aruco
        self._dict = _aruco_dictionary()
        p = a.DetectorParameters() if hasattr(a, 'DetectorParameters') and not hasattr(a, 'DetectorParameters_create') \
            else a.DetectorParameters_create()
        # small, slanted tags: look at small quads, several threshold windows,
        # and refine the corners to sub-pixel. Set one by one so a parameter
        # that a given OpenCV version does not have is skipped, not fatal.
        for name, value in (('adaptiveThreshWinSizeMin', 5), ('adaptiveThreshWinSizeMax', 23),
                            ('adaptiveThreshWinSizeStep', 6), ('minMarkerPerimeterRate', 0.012),
                            ('maxMarkerPerimeterRate', 4.0), ('polygonalApproxAccuracyRate', 0.05),
                            ('minCornerDistanceRate', 0.02), ('minMarkerDistanceRate', 0.02),
                            ('cornerRefinementMethod', a.CORNER_REFINE_SUBPIX),
                            ('cornerRefinementWinSize', 4), ('cornerRefinementMaxIterations', 40),
                            ('cornerRefinementMinAccuracy', 0.01), ('perspectiveRemovePixelPerCell', 8),
                            ('errorCorrectionRate', 0.0)):          # 0.0 = like max_hamming 0
            try:
                setattr(p, name, value)
            except Exception:
                pass
        self._params = p
        self._aruco = a.ArucoDetector(self._dict, p) if hasattr(a, 'ArucoDetector') else None

    def _detect_aruco(self, gray):
        img = gray
        s = 1.0
        if self.decimate > 1.0:
            s = 1.0 / self.decimate
            img = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if self._aruco is not None:
            corners, ids, _ = self._aruco.detectMarkers(img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(img, self._dict, parameters=self._params)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            c = np.asarray(c, float).reshape(4, 2) / s
            # OpenCV's 36h11 bitmaps are the official AprilTag images rotated by
            # 180 deg, so its "top-left" is the official tag's bottom-right
            out.append(TagDetection(int(i), c[[2, 3, 0, 1]], margin=100.0))
        return out

    # ---- AprilTag backend ----
    def _detect_apriltag(self, gray):
        out = []
        for d in self._at.detect(np.ascontiguousarray(gray)):
            if d.hamming != 0 or d.decision_margin < self.min_margin:
                continue
            # The AprilTag library puts pixel centres at (i + 0.5, j + 0.5); OpenCV,
            # and therefore the camera calibration, puts them at (i, j). Half a
            # pixel is 5 mm on the floor at the far end of the arena.
            c = np.asarray(d.corners, float) - APRILTAG_PIXEL_OFFSET  # lb, rb, rt, lt
            out.append(TagDetection(d.tag_id, c[[3, 2, 1, 0]], margin=d.decision_margin))
        return out

    def detect(self, gray, roi=None):
        """gray: uint8 image. roi: optional (x0, y0, x1, y1) to search only part
        of the frame (corners come back in full-frame pixels)."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        ox = oy = 0
        if roi is not None:
            x0, y0, x1, y1 = [int(v) for v in roi]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
            if x1 - x0 < 16 or y1 - y0 < 16:
                return []
            gray = gray[y0:y1, x0:x1]
            ox, oy = x0, y0
        dets = self._detect_apriltag(gray) if self.backend == 'apriltag' else self._detect_aruco(gray)
        keep = {}
        for d in dets:
            if ox or oy:
                d.corners = d.corners + [ox, oy]
                d.center = d.center + [ox, oy]
            if d.min_side_px() < self.min_side_px:
                continue
            if d.id not in keep or d.margin > keep[d.id].margin:  # one detection per id
                keep[d.id] = d
        return list(keep.values())


def corners_from_apriltag_msg(corner_points):
    """apriltag_msgs/AprilTagDetection.corners (lb, rb, rt, lt) -> tl, tr, br, bl."""
    c = np.array([[p.x, p.y] for p in corner_points], float) - APRILTAG_PIXEL_OFFSET
    return c[[3, 2, 1, 0]]


# --------------------------------------------------------------------------
#  Tag on a known plane -> pose on the floor
# --------------------------------------------------------------------------

def tag_floor_pose(det, floor_map, tag_height, tag_size=None):
    """Arena pose of a tag lying flat at `tag_height` above the floor.

    Returns dict(x, y, yaw, size, quality) or None. yaw is the direction, in
    the arena, of the tag's "up" edge (bottom -> top of the printed tag). size
    is the tag's measured edge length on the plane; comparing it with the real
    edge length is a strong sanity check on the whole chain (calibration, tag
    height, detection), so a tag whose measured size is off by more than 20 % is
    rejected when tag_size is given.
    """
    P = floor_map.pixel_to_plane(det.corners, tag_height)
    if not np.all(np.isfinite(P)):
        return None
    tl, tr, br, bl = P
    center = P.mean(0)
    up = (tl + tr) / 2 - (bl + br) / 2
    right = (tr + br) / 2 - (tl + bl) / 2
    # the two axes give two estimates of yaw 90 deg apart; average them
    y1 = math.atan2(up[1], up[0])
    y2 = math.atan2(right[1], right[0]) + math.pi / 2
    yaw = math.atan2(math.sin(y1) + math.sin(y2), math.cos(y1) + math.cos(y2))
    sides = [np.linalg.norm(tl - tr), np.linalg.norm(tr - br), np.linalg.norm(br - bl), np.linalg.norm(bl - tl)]
    size = float(np.mean(sides))
    skew = float(np.ptp(sides) / max(size, 1e-9))                 # 0 for a perfect square
    # a mirrored quad means the corners are inconsistent with a tag seen from above
    cross = up[0] * right[1] - up[1] * right[0]
    if cross >= 0:                       # for a tag seen from above, right is clockwise of up
        return None
    if tag_size is not None and tag_size > 0:
        if abs(size - tag_size) > 0.20 * tag_size or skew > 0.35:
            return None
    return dict(x=float(center[0]), y=float(center[1]), yaw=float(yaw), size=size, skew=skew)
