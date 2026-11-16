"""Floor calibration (where is the camera?) and mount planning (where should it
be?). Pure functions; tools/swarm_calibrate.py and tools/swarm_plan_mount.py
are the command-line front ends."""
import math

import numpy as np

from .camera import FloorMap
from .tags import quad_center


def average_detections(frames):
    """frames: list of dict tag_id -> TagDetection (one dict per frame).
    Returns dict tag_id -> (mean corners (4, 2), n frames, corner jitter px).
    Averaging 20-30 frames of a still scene takes the corner noise out of the
    calibration."""
    acc = {}
    for f in frames:
        for i, d in f.items():
            acc.setdefault(i, []).append(d.corners)
    out = {}
    for i, lst in acc.items():
        a = np.stack(lst)
        jitter = float(np.sqrt(((a - a.mean(0)) ** 2).sum(2).mean())) if len(a) > 1 else 0.0
        out[i] = (a.mean(0), len(a), jitter)
    return out


def anchor_points(mean_corners, anchors):
    """Build the pixel / world correspondences.

    anchors: dict tag_id -> dict(x, y, z=0.0, yaw=None, size=None). An anchor
    with only x, y contributes its centre. An anchor that also gives yaw (deg,
    direction of the tag's top edge in the arena) and size contributes its four
    corners too, which makes each tag worth five points; only use that if the
    tag really was laid down square to the arena axes."""
    px, W, ids = [], [], []
    for i, a in anchors.items():
        if i not in mean_corners:
            continue
        c = mean_corners[i][0]
        z = float(a.get('z', 0.0))
        px.append(quad_center(c))
        W.append([float(a['x']), float(a['y']), z])
        ids.append(i)
        if a.get('yaw') is not None and a.get('size'):
            yaw = math.radians(float(a['yaw']))
            h = float(a['size']) / 2
            up = np.array([math.cos(yaw), math.sin(yaw)])
            rt = np.array([math.sin(yaw), -math.cos(yaw)])
            ctr = np.array([float(a['x']), float(a['y'])])
            for k, (su, sr) in enumerate(((1, -1), (1, 1), (-1, 1), (-1, -1))):   # tl, tr, br, bl
                w = ctr + su * h * up + sr * h * rt
                px.append(c[k])
                W.append([w[0], w[1], z])
                ids.append(i)
    return np.array(px, float).reshape(-1, 2), np.array(W, float).reshape(-1, 3), ids


def calibrate_floor(camera, mean_corners, anchors):
    """Solve the camera pose and grade the result.

    Returns (FloorMap, report). report['grade'] is 'good', 'usable' or 'bad',
    and report['problems'] says what to fix. Leave-one-out error (solve
    without each anchor, predict where it is) is the honest accuracy number:
    the fit residual alone always looks good with few points."""
    px, W, ids = anchor_points(mean_corners, anchors)
    used = sorted(set(ids))
    problems = []
    if len(used) < 4:
        raise ValueError('only %d of the %d reference tags were seen (%s); need at least 4. Seen tag ids: %s'
                         % (len(used), len(anchors), used, sorted(mean_corners)))
    # spread: points along a line cannot fix the camera pose
    xy = np.array([[anchors[i]['x'], anchors[i]['y']] for i in used], float)
    sv = np.linalg.svd(xy - xy.mean(0), compute_uv=False)
    if sv[1] < 0.15 * sv[0] or sv[1] < 0.25:
        problems.append('reference tags are nearly in a line (spread %.2f m by %.2f m); '
                        'spread them over the whole floor, including the far side' % (sv[0], sv[1]))
    fm = FloorMap.solve(camera, px, W)
    rep = dict(fm.info)
    rep['anchors_used'] = used
    rep['anchors_missing'] = sorted(set(anchors) - set(used))
    loo = []
    if len(used) >= 5:
        ids_a = np.array(ids)
        for i in used:
            keep = ids_a != i
            try:
                f2 = FloorMap.solve(camera, px[keep], W[keep])
            except Exception:
                continue
            a = anchors[i]
            p = f2.pixel_to_plane(mean_corners[i][0], float(a.get('z', 0.0))).mean(0)
            loo.append(float(math.hypot(p[0] - a['x'], p[1] - a['y'])))
    rep['leave_one_out_m'] = loo
    rep['leave_one_out_rms_m'] = float(np.sqrt(np.mean(np.square(loo)))) if loo else None
    cz = fm.center[2]
    if cz < 0.3:
        problems.append('solved camera height is %.2f m; the reference coordinates are probably '
                        'mirrored or in the wrong order' % cz)
    worst = rep['leave_one_out_rms_m'] if loo else rep['floor_rms_m'] * 2
    if rep['reproj_rms_px'] > 3.0:
        problems.append('reprojection error %.1f px: a reference position is mistyped or a tag id is '
                        'swapped (check the per-point errors)' % rep['reproj_rms_px'])
    if not loo:
        problems.append('fewer than 5 reference tags: accuracy cannot be cross-checked; use 6 or more')
    rep['grade'] = 'bad' if (worst > 0.05 or rep['reproj_rms_px'] > 3.0 or cz < 0.3) else \
        ('usable' if (worst > 0.02 or problems) else 'good')
    rep['problems'] = problems
    return fm, rep


# --------------------------------------------------------------------------
#  Mount planning
# --------------------------------------------------------------------------

def coverage(floor_map, bounds, tag_size=0.15, tag_height=0.12, step=0.25, margin_px=12, pixel_sigma=0.35):
    """What a camera pose is worth over the arena rectangle
    bounds = (x0, x1, y0, y1). For every grid point: is a whole tag inside the
    image, how many pixels does one tag cell cover on its short (foreshortened)
    axis, and what position sigma does that give. Returns dict of arrays."""
    x0, x1, y0, y1 = bounds
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    vis = np.zeros((len(ys), len(xs)), bool)
    cell = np.zeros((len(ys), len(xs)))
    sig = np.full((len(ys), len(xs)), np.nan)
    h = tag_size / 2
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            worst = np.inf
            ok = True
            for yaw in (0.0, math.pi / 4):
                c, s = math.cos(yaw), math.sin(yaw)
                cs = [[x + a * h * c - b * h * s, y + a * h * s + b * h * c, tag_height]
                      for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
                px, front = floor_map.world_to_pixel(cs)
                if not front.all() or not floor_map.in_image(px, margin_px).all():
                    ok = False
                    break
                sides = [np.linalg.norm(px[k] - px[(k + 1) % 4]) for k in range(4)]
                # short axis of the projected quad: distance between opposite edge midpoints
                m = [np.linalg.norm((px[0] + px[1]) / 2 - (px[2] + px[3]) / 2),
                     np.linalg.norm((px[1] + px[2]) / 2 - (px[3] + px[0]) / 2)]
                worst = min(worst, min(m) / 8.0, min(sides) / 8.0 * 1.0)
            vis[j, i] = ok
            if ok:
                cell[j, i] = worst
                _, along = floor_map.ground_sample((x, y), tag_height)
                sig[j, i] = pixel_sigma * along
    return dict(xs=xs, ys=ys, visible=vis, px_per_cell=cell, sigma_m=sig)


GOOD_PX_PER_CELL = 2.5      # comfortable for the AprilTag detector at full resolution
MIN_PX_PER_CELL = 1.8       # below this, detections drop out with any blur or glare


def score_coverage(cov):
    vis = cov['visible']
    frac = float(vis.mean())
    cell = cov['px_per_cell']
    mn = float(cell[vis].min()) if vis.any() else 0.0
    ok = float(((cell >= MIN_PX_PER_CELL) & vis).mean())
    good = float(((cell >= GOOD_PX_PER_CELL) & vis).mean())
    sig = cov['sigma_m']
    return dict(visible_fraction=frac, usable_fraction=ok, good_fraction=good, min_px_per_cell=mn,
                worst_sigma_mm=float(np.nanmax(sig) * 1000) if vis.any() else float('nan'),
                median_sigma_mm=float(np.nanmedian(sig) * 1000) if vis.any() else float('nan'))


def best_aim(camera, position, bounds, tag_size=0.15, tag_height=0.12, step=0.25, rolls=(0.0,)):
    """Search where to point a camera fixed at `position` so that as much of
    the arena as possible is usable, then as sharp as possible. Returns
    (FloorMap, look_at, roll_deg, score)."""
    x0, x1, y0, y1 = bounds
    best = None
    for roll in rolls:
        for fx in np.linspace(0.15, 0.85, 8):
            for fy in np.linspace(0.15, 0.85, 8):
                look = (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), 0.0)
                if math.hypot(look[0] - position[0], look[1] - position[1]) < 0.2:
                    continue
                fm = FloorMap.from_mount(camera, position, look, roll)
                sc = score_coverage(coverage(fm, bounds, tag_size, tag_height, step=max(step, 0.3)))
                key = (round(sc['usable_fraction'], 3), round(sc['good_fraction'], 3), sc['min_px_per_cell'])
                if best is None or key > best[0]:
                    best = (key, fm, look, roll, sc)
    return best[1], best[2], best[3], best[4]
