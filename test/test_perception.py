"""End-to-end checks of the perception chain on rendered frames from a
corner-mounted camera with the lab's real lens calibration."""
import math
import os

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from swarm_cps.core.camera import CameraModel, FloorMap, trilaterate
from swarm_cps.core.calib import average_detections, calibrate_floor, coverage, score_coverage, best_aim
from swarm_cps.core.localize import Localizer, PoseFilter, RobotTag, wrap
from swarm_cps.core.synth import Scene
from swarm_cps.core.tags import Detector, tag_floor_pose, tag_image

CAL = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_cam.yaml')
ROOM = (0.0, 4.2, 0.0, 3.2)                       # arena x0, x1, y0, y1 (m)
CAM_POS = (-0.15, -0.15, 2.3)                     # in a top corner of the room
ANCHORS = {100: (0.6, 0.5), 101: (3.8, 0.5), 102: (3.8, 2.8), 103: (0.6, 2.8), 104: (2.2, 1.6), 105: (2.2, 0.4),
           106: (1.2, 2.2)}


def have_apriltag():
    try:
        import pupil_apriltags  # noqa: F401
        return True
    except Exception:
        return False


BACKENDS = ['aruco'] + (['apriltag'] if have_apriltag() else [])


@pytest.fixture(scope='module')
def cam():
    return CameraModel.from_ros_yaml(CAL)


@pytest.fixture(scope='module')
def true_map(cam):
    return FloorMap.from_mount(cam, CAM_POS, (2.0, 1.5, 0.0))


@pytest.fixture(scope='module')
def frame(true_map):
    """One rendered frame: seven floor reference tags and four rovers."""
    sc = Scene(true_map, seed=4)
    for i, (x, y) in ANCHORS.items():
        sc.add_tag(i, x, y, 0.3 * i, z=0.0, size=0.15)
    rovers = {1: (1.0, 1.0, 0.4), 2: (2.8, 1.2, -2.0), 3: (3.3, 2.3, 2.6), 4: (1.6, 2.6, -0.7)}
    for i, (x, y, yaw) in rovers.items():
        sc.add_tag(i, x, y, yaw, z=0.12, size=0.15)
    return sc.render(ss=2), rovers


def test_lens_model_inverts_over_the_whole_frame(cam):
    g = np.array([[x, y] for x in np.linspace(0, 1919, 40) for y in np.linspace(0, 1079, 24)])
    assert cam.undistort_residual_px(g) < 1e-6
    xn = cam.undistort(g)
    pp, _ = cv2.projectPoints(np.concatenate([xn, np.ones((len(xn), 1))], 1).reshape(-1, 1, 3),
                              np.zeros(3), np.zeros(3), cam.K, cam.D)
    assert np.abs(pp.reshape(-1, 2) - g).max() < 1e-6          # agrees with OpenCV's own model
    hf, vf = cam.fov_deg()
    assert 100 < hf < 120 and 55 < vf < 68                     # the real lens is ~111 x 61 deg


def test_pose_from_floor_points_recovers_the_camera(cam, true_map):
    rng = np.random.default_rng(0)
    W = np.array([[x, y, 0.0] for x in (0.5, 2.0, 3.8) for y in (0.4, 1.6, 2.9)])
    px, front = true_map.world_to_pixel(W)
    assert front.all()
    fm = FloorMap.solve(cam, px + rng.normal(0, 0.2, px.shape), W)
    assert np.linalg.norm(fm.center - true_map.center) < 0.02
    # and it maps a point at tag height correctly, which is what matters
    q, _ = true_map.world_to_pixel([[3.0, 2.0, 0.12]])
    assert np.linalg.norm(fm.pixel_to_plane(q, 0.12)[0] - [3.0, 2.0]) < 0.004


def test_pose_solver_rejects_too_few_points(cam):
    with pytest.raises(ValueError):
        FloorMap.solve(cam, [[1, 2], [3, 4], [5, 6]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_tag_bitmap_matches_the_apriltag_library():
    """OpenCV stores the 36h11 family rotated 180 deg from the official images.
    tag_image() must hand out the official orientation, or printed tags and
    headings would disagree between the two detector backends."""
    if not have_apriltag():
        pytest.skip('pupil_apriltags not installed')
    img = cv2.copyMakeBorder(tag_image(7, 20, 2), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=128)
    a = Detector('apriltag').detect(img)[0]
    b = Detector('aruco').detect(img)[0]
    assert a.id == b.id == 7
    assert np.abs(a.corners - b.corners).max() < 1.5           # same corner order, same place
    assert a.corners[0][0] < a.corners[1][0] and a.corners[0][1] < a.corners[3][1]   # tl is top-left


@pytest.mark.parametrize('backend', BACKENDS)
def test_calibrate_then_localize(cam, frame, backend):
    """The Wednesday procedure, on a rendered frame: detect the floor
    reference tags, solve the camera pose from them, then locate the rovers."""
    img, rovers = frame
    det = Detector(backend, decimate=1.0, threads=2)
    found = {d.id: d for d in det.detect(img)}
    anchors = {i: dict(x=x, y=y) for i, (x, y) in ANCHORS.items()}
    fm, rep = calibrate_floor(cam, average_detections([found]), anchors)
    assert rep['grade'] in ('good', 'usable'), rep
    assert rep['leave_one_out_rms_m'] < 0.01
    assert abs(fm.center[2] - CAM_POS[2]) < 0.03
    seen = 0
    for i, (x, y, yaw) in rovers.items():
        if i not in found:
            continue
        p = tag_floor_pose(found[i], fm, 0.12, 0.15)
        assert p is not None
        assert math.hypot(p['x'] - x, p['y'] - y) < 0.012
        assert abs(wrap(p['yaw'] - yaw)) < math.radians(2.5)
        assert abs(p['size'] - 0.15) < 0.01
        seen += 1
    assert seen >= (4 if backend == 'apriltag' else 3)


def test_wrong_tag_height_is_caught_by_the_size_check(cam, frame, true_map):
    img, rovers = frame
    det = Detector(BACKENDS[-1], threads=2)
    d = {x.id: x for x in det.detect(img)}[1]
    assert tag_floor_pose(d, true_map, 0.12, 0.15) is not None
    assert tag_floor_pose(d, true_map, 0.80, 0.15) is None      # tag "at 0.8 m" would be far too small


def test_calibration_refuses_collinear_or_missing_anchors(cam, frame):
    img, _ = frame
    det = Detector(BACKENDS[-1], threads=2)
    found = {d.id: d for d in det.detect(img)}
    with pytest.raises(ValueError):
        calibrate_floor(cam, average_detections([found]), {100: dict(x=0.6, y=0.5), 101: dict(x=3.8, y=0.5)})
    # a mistyped coordinate (tag 104 entered 1 m off) must not grade 'good'
    bad = {i: dict(x=x, y=y) for i, (x, y) in ANCHORS.items()}
    assert 104 in found
    bad[104] = dict(x=3.2, y=1.6)                               # really at (2.2, 1.6)
    _, rep = calibrate_floor(cam, average_detections([found]), bad)
    assert rep['grade'] == 'bad'


def test_filter_estimates_speed_and_rejects_outliers():
    f = PoseFilter()
    rng = np.random.default_rng(1)
    t, x = 0.0, 0.0
    for k in range(90):                                         # 3 s at 30 Hz, 0.14 m/s along +x
        t += 1 / 30
        x += 0.14 / 30
        assert f.update(t, x + rng.normal(0, 0.002), 1.0 + rng.normal(0, 0.002), rng.normal(0, 0.01), 0.003)
    assert abs(f.forward_speed - 0.14) < 0.03
    v0 = f.forward_speed
    assert not f.update(t + 1 / 30, x + 0.8, 1.0, 0.0, 0.003)   # a wild detection is ignored...
    assert abs(f.forward_speed - v0) < 1e-9                     # ...and does not kick the velocity
    assert not f.update(t, x, 1.0, 0.0, 0.003)                  # repeated stamp
    assert not f.update(t - 0.5, x, 1.0, 0.0, 0.003)            # reordered stamp
    # but a real jump (rover picked up and moved) is accepted after a few frames
    ok = [f.update(t + (k + 2) / 30, 3.0, 2.0, 1.0, 0.003) for k in range(6)]
    assert ok[-1] and math.hypot(f.pose[0] - 3.0, f.pose[1] - 2.0) < 0.01


def test_tracking_window_follows_a_moving_rover(cam, true_map):
    """Windowed search must give the same poses as a full search, and must
    actually be used (that is where the frame rate comes from)."""
    det = Detector(BACKENDS[-1], threads=2)
    robots = [RobotTag('rover1', 1), RobotTag('rover2', 2, yaw_offset=math.pi / 2, offset=(-0.05, 0.0))]
    loc = Localizer(det, true_map, robots, rescan_period=100.0)
    errs = []
    for k in range(8):
        t = k / 15.0
        x1, y1, a1 = 1.2 + 0.25 * t * 4, 1.0 + 0.1 * t, 0.2
        x2, y2, a2 = 3.0 - 0.2 * t * 4, 2.2, 2.9
        sc = Scene(true_map, seed=k)
        sc.add_tag(1, x1, y1, a1, z=0.12)
        sc.add_tag(2, x2, y2, a2, z=0.12)
        out = loc.process(sc.render(ss=1), t)
        assert set(out) == {'rover1', 'rover2'}
        errs.append(math.hypot(out['rover1']['raw'][0] - x1, out['rover1']['raw'][1] - y1))
        # rover2's tag is mounted rotated by -90 deg and 5 cm ahead of the reference point
        yaw2 = wrap(a2 + math.pi / 2)
        ex = x2 - 0.05 * math.cos(yaw2)
        ey = y2 - 0.05 * math.sin(yaw2)
        assert math.hypot(out['rover2']['raw'][0] - ex, out['rover2']['raw'][1] - ey) < 0.012
        assert abs(wrap(out['rover2']['raw'][2] - yaw2)) < math.radians(3)
    assert max(errs) < 0.012
    assert loc.stats['full_scans'] == 1 and loc.stats['roi_hits'] == 7


def test_one_rover_out_of_view_does_not_slow_the_others_to_full_search_speed(cam, true_map):
    det = Detector(BACKENDS[-1], threads=2)
    loc = Localizer(det, true_map, [RobotTag('rover1', 1), RobotTag('rover2', 2)], rescan_period=100.0, lost_rescan=0.2)
    sc = Scene(true_map, seed=1)
    sc.add_tag(1, 1.5, 1.2, 0.3, z=0.12)                       # rover2 is not on the floor at all
    img = sc.render(ss=1)
    seen = 0
    for k in range(30):                                         # one second at 30 Hz
        seen += 'rover1' in loc.process(img, k / 30.0)
    assert seen == 30
    assert loc.stats['full_scans'] <= 7                         # about 5 per second, not 30


def test_trilateration():
    p = trilaterate(math.hypot(2, 1.5), math.hypot(1, 1.5), (0, 0), (3, 0))
    assert np.allclose(p, [2, 1.5], atol=1e-9)
    assert np.allclose(trilaterate(math.hypot(2, 1.5), math.hypot(1, 1.5), (0, 0), (3, 0), side=-1), [2, -1.5])
    with pytest.raises(ValueError):
        trilaterate(0.5, 0.5, (0, 0), (3, 0))


def test_mount_planner_prefers_a_sensible_aim(cam):
    fm, look, roll, sc = best_aim(cam, CAM_POS, ROOM, step=0.4)
    assert sc['visible_fraction'] > 0.9
    d = fm.describe()
    assert 25 < d['pitch_down_deg'] < 65                        # looking across and down, not at the wall
    # looking straight down from the corner sees much less of the floor
    down = FloorMap.from_mount(cam, CAM_POS, (CAM_POS[0] + 0.01, CAM_POS[1] + 0.01, 0.0))
    assert score_coverage(coverage(down, ROOM, step=0.4))['visible_fraction'] < sc['visible_fraction']
