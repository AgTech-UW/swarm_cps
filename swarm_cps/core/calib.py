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
#  Calibration from cards thrown on the floor (nothing measured but the tag size)
# --------------------------------------------------------------------------

def _rodrigues(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _square(x, y, yaw, size):
    """Floor corners (tl, tr, br, bl) of a flat tag whose top edge faces `yaw`."""
    h = size / 2
    up = np.array([math.cos(yaw), math.sin(yaw)])
    rt = np.array([math.sin(yaw), -math.cos(yaw)])
    c = np.array([x, y])
    return np.array([c + h * up - h * rt, c + h * up + h * rt, c - h * up + h * rt, c - h * up - h * rt])


def calibrate_floor_auto(camera, mean_corners, size, origin=None, xaxis=None, origin_at=(0.0, 0.0), exclude=()):
    """Camera pose from tags lying anywhere on the floor, none of them surveyed.

    Every card is a square of known size on one plane. That is enough to
    recover the camera's height and tilt and where every card lies relative
    to the others: the unknowns are the camera pose and each card's (x, y,
    yaw), and they are solved together by least squares on the corner pixels.
    The arena frame is then pinned by two cards: `origin` sits at `origin_at`,
    and +x points from it toward `xaxis` (without `xaxis`, +x is the origin
    card's own left-to-right direction, which is less accurate).

    Scale comes entirely from `size`, so measure a printed card: 1 mm wrong on
    a 150 mm card is 3 cm over 4 m. Returns (FloorMap, anchors, report), with
    anchors = dict id -> dict(x, y, z=0) of where the cards were found."""
    from .camera import _rvec_from_rot
    ids = sorted(i for i in mean_corners if i not in set(exclude))
    if len(ids) < 4:
        raise ValueError('only %d floor cards seen (%s); throw down at least 5, spread out' % (len(ids), ids))
    if origin is None:
        origin = ids[0]
    if origin not in ids:
        raise ValueError('origin card %s is not in view (seen: %s)' % (origin, ids))
    if xaxis is not None and xaxis not in ids:
        raise ValueError('x-axis card %s is not in view (seen: %s)' % (xaxis, ids))
    xn = {i: camera.undistort(mean_corners[i][0]) for i in ids}

    def side(i):
        c = mean_corners[i][0]
        return min(np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4))
    ref = max(ids, key=side)                              # start from the card that is biggest in the image
    fm = FloorMap.solve(camera, mean_corners[ref][0], np.c_[_square(0.0, 0.0, math.pi / 2, size), np.zeros(4)], refine=False)
    others = [i for i in ids if i != ref]
    p0 = list(_rvec_from_rot(fm.R)) + list(fm.t)
    for i in others:                                      # where the first guess puts every other card
        P = fm.pixel_to_plane(mean_corners[i][0], 0.0)
        if not np.all(np.isfinite(P)):
            raise ValueError('card %d cannot be placed on the floor from the first guess; is it lying flat?' % i)
        up = (P[0] + P[1]) / 2 - (P[2] + P[3]) / 2
        p0 += [float(P[:, 0].mean()), float(P[:, 1].mean()), math.atan2(up[1], up[0])]
    p = np.array(p0, float)
    ref_sq = np.c_[_square(0.0, 0.0, math.pi / 2, size), np.zeros(4)]

    def residual(q):
        R, t = _rodrigues(q[:3]), q[3:6]
        out = []
        for k, i in enumerate([ref] + others):
            if k == 0:
                Wd = ref_sq
            else:
                x, y, yaw = q[6 + 3 * (k - 1): 9 + 3 * (k - 1)]
                Wd = np.c_[_square(x, y, yaw, size), np.zeros(4)]
            Xc = Wd @ R.T + t
            z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
            out.append((Xc[:, :2] / z[:, None] - xn[i]).ravel())
        return np.concatenate(out)

    f = float(camera.K[0, 0])
    lam = 1e-3
    r = residual(p)
    cost = float(r @ r)
    for _ in range(80):                                   # Levenberg-Marquardt, numeric Jacobian
        J = np.empty((len(r), len(p)))
        for j in range(len(p)):
            dq = np.zeros(len(p))
            dq[j] = 1e-6
            J[:, j] = (residual(p + dq) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(8):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            r2 = residual(p + step)
            c2 = float(r2 @ r2)
            if c2 < cost:
                small = (cost - c2) < 1e-14 * max(cost, 1e-30) or np.abs(step).max() < 1e-10
                p, r, cost, lam, improved = p + step, r2, c2, max(lam / 5, 1e-9), True
                break
            lam *= 5
        if not improved or small:
            break
    R, t = _rodrigues(p[:3]), p[3:6]
    cards = {ref: (0.0, 0.0, math.pi / 2)}
    for k, i in enumerate(others):
        cards[i] = tuple(float(v) for v in p[6 + 3 * k: 9 + 3 * k])
    if (-R.T @ t)[2] < 0:
        raise ValueError('solution puts the camera under the floor; a card id is probably duplicated or misread')
    # pin the arena frame to the origin (and x-axis) cards
    ox, oy, oyaw = cards[origin]
    if xaxis is not None:
        th = math.atan2(cards[xaxis][1] - oy, cards[xaxis][0] - ox)
        base = math.hypot(cards[xaxis][0] - ox, cards[xaxis][1] - oy)
    else:
        th = oyaw - math.pi / 2                           # the card's left-to-right direction
        base = None
    c, s_ = math.cos(-th), math.sin(-th)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])   # old frame -> new frame rotation
    o_new = np.array([origin_at[0], origin_at[1], 0.0])
    c_old = np.array([ox, oy, 0.0])
    R_new = R @ Rz.T
    t_new = R @ (c_old - Rz.T @ o_new) + t
    fm = FloorMap(camera, R_new, t_new)
    anchors = {}
    for i, (x, y, yaw) in cards.items():
        q = Rz @ (np.array([x, y, 0.0]) - c_old) + o_new
        anchors[i] = dict(x=float(q[0]), y=float(q[1]), z=0.0)
    # how good is it
    px_err = np.sqrt(np.mean(r.reshape(-1, 2) ** 2 * 1.0)) * f * math.sqrt(2)
    sizes = []
    for i in ids:
        P = fm.pixel_to_plane(mean_corners[i][0], 0.0)
        sizes.append(float(np.mean([np.linalg.norm(P[k] - P[(k + 1) % 4]) for k in range(4)])))
    size_err = float(np.max(np.abs(np.array(sizes) - size)) / size)
    xy = np.array([[a['x'], a['y']] for a in anchors.values()])
    sv = np.linalg.svd(xy - xy.mean(0), compute_uv=False)
    problems = []
    if len(ids) < 6:
        problems.append('only %d cards: use 6 or more for a result you can trust' % len(ids))
    if sv[1] < 0.2 * sv[0] or sv[1] < 0.4:
        problems.append('the cards are nearly in a line (spread %.1f m by %.1f m); scatter them over the whole floor'
                        % (sv[0], sv[1]))
    if sv[0] < 1.5:
        problems.append('the cards span only %.1f m; put some at the far side of the floor' % sv[0])
    if xaxis is None:
        problems.append('no x-axis card given: +x comes from how card %d happens to be turned' % origin)
    elif base < 1.0:
        problems.append('origin and x-axis cards are only %.2f m apart; put them 2 m or more apart' % base)
    if px_err > 1.5:
        problems.append('corner fit is %.1f px: a card is not flat, is curled, or its size differs from %.0f mm'
                        % (px_err, 1e3 * size))
    grade = 'bad' if (px_err > 3.0 or size_err > 0.08) else ('usable' if problems else 'good')
    rep = dict(reproj_rms_px=float(px_err), size_max_error=size_err, cards_used=ids, n_points=4 * len(ids),
               origin=origin, xaxis=xaxis, spread_m=[float(sv[0]), float(sv[1])], problems=problems, grade=grade,
               floor_rms_m=float(np.sqrt(np.mean((np.array(sizes) - size) ** 2))), auto=True)
    fm.info = dict(reproj_rms_px=float(px_err), n_points=4 * len(ids), auto=True)
    return fm, anchors, rep


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
