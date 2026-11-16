"""Draw the route the leader follows (followers keep formation behind it).

    draw_path                                  # opens a window: click waypoints on the floor
    draw_path --points 0.6,1.6 2.0,1.6 3.4,2.4 # no window: type them (x,y in metres)

In the window: left click adds a waypoint, right click or "u" removes the
last, "s" saves, "q" quits without saving. The first point is where the leader
starts, facing the second. The grey line is what will actually be driven:
waypoints joined by curves no tighter than mission.turn_radius. A loop in the
grey line means two waypoints are too close for the rover to turn between;
move them apart. Saving rewrites the mission in arena.yaml (old file kept as
arena.yaml.bak); bounds, walls, rocks and zones are untouched.
Then:  simulate --check
"""
import argparse
import math
import shutil
import sys

import numpy as np

from ..core.arena import Arena
from ..core.controller import SwarmController
from ..core.layout import dump_yaml
from .common import cv2, say, settings


def legs_from_points(pts):
    """Waypoints -> (start pose, legs). Each waypoint is passed heading along
    the line from the one before it to the one after it."""
    pts = [(float(x), float(y)) for x, y in pts]
    if len(pts) < 2:
        raise ValueError('need at least two points')
    for a, b in zip(pts, pts[1:]):
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 0.15:
            raise ValueError('two waypoints are less than 15 cm apart')

    def heading(i):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        return round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), 1)
    start = [round(pts[0][0], 2), round(pts[0][1], 2), round(math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])), 1)]
    legs = [dict(name='leg %d' % i, to=[round(pts[i][0], 2), round(pts[i][1], 2), heading(i)], work=True)
            for i in range(1, len(pts))]
    return start, legs


def build(st, pts):
    """Config with the new mission, the Arena made from it, and its problems."""
    import yaml
    with open(st.arena_path) as f:
        cfg = yaml.safe_load(f)
    start, legs = legs_from_points(pts)
    m = cfg['mission']
    n = len(st.names)
    m.update(start=start, legs=legs, lead_in=round((n - 1) * st.params().spacing + 1.0, 2))
    m.setdefault('swath', 0.3)
    m.setdefault('turn_radius', 0.55)
    m.pop('solo', None)
    for k in ('walls', 'obstacles', 'zones'):
        cfg[k] = cfg.get(k) or []
    cfg.setdefault('compass', dict(plus_x='east', plus_y='north'))
    cfg.setdefault('edge_margin', 0.2)
    ar = Arena(cfg)
    p = st.params()
    ctl = SwarmController(ar, st.names, p)
    probs = ar.validate(ctl.route, p.body_radius, p.min_turn_radius)
    straight = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    if ctl.route.length > 1.5 * straight + 0.5:
        probs.append('the driven line is %.1f m for %.1f m of waypoints: it loops because waypoints are too close '
                     'together for a %.2f m turn' % (ctl.route.length, straight, ar.turn_radius))
    for nme, (x, y, _) in ctl.start_poses().items():
        if not ar.inside(x, y, 0.12):
            probs.append('%s would start off the floor at (%.2f, %.2f): move the first point further in' % (nme, x, y))
    return cfg, ar, ctl, probs


def save(st, cfg):
    shutil.copy(st.arena_path, st.arena_path + '.bak')
    with open(st.arena_path, 'w') as f:
        f.write(dump_yaml(cfg, 'The floor and the mission. Route drawn with draw_path.\nCheck it with:  simulate --check'))
    say('wrote %s (previous version: arena.yaml.bak)' % st.arena_path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='draw_path', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config-dir', dest='config_dir', default=None)
    ap.add_argument('--points', nargs='+', default=None, help='waypoints as x,y x,y ... instead of the window')
    args = ap.parse_args(argv)
    st = settings(args)
    if args.points:
        try:
            pts = [tuple(float(v) for v in p.split(',')) for p in args.points]
            cfg, ar, ctl, probs = build(st, pts)
        except Exception as e:
            say('PATH: %s' % e)
            return 1
        for pr in probs:
            say('  [FAIL] ' + pr)
        if probs:
            say('not saved')
            return 1
        save(st, cfg)
        say('route %.1f m. Now run:  simulate --check' % ctl.route.length)
        return 0
    from ..core.viz import TopDown
    base = st.arena()
    td = TopDown(base, px_per_m=170, panel=360)
    pts, state = [], dict(dirty=True, cfg=None, probs=['click at least two points'])

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and x < td.w:
            wx = x / td.k + base.x0 - td.pad
            wy = (td.h - y) / td.k + base.y0 - td.pad
            pts.append((round(wx, 2), round(wy, 2)))
            state['dirty'] = True
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()
            state['dirty'] = True
    cv2.namedWindow('draw the route')
    cv2.setMouseCallback('draw the route', click)
    img = None
    while True:
        if state['dirty']:
            state['dirty'] = False
            ctl = None
            state['cfg'], state['probs'] = None, ['click at least two points']
            if len(pts) >= 2:
                try:
                    state['cfg'], _, ctl, state['probs'] = build(st, pts)
                except Exception as e:
                    state['probs'] = [str(e)]
            td.trails = {}
            lines = ['left click: add   right click / u: undo', 's: save   q: quit', ''] + \
                    (['PROBLEM: ' + p for p in state['probs']] if state['probs'] else ['route ok, %.1f m' % ctl.route.length])
            wrapped = []
            for ln in lines:
                wrapped += [ln[i:i + 38] for i in range(0, max(len(ln), 1), 38)]
            img = td.draw(ctl, ctl.start_poses() if ctl else None, title='draw the route', extra=wrapped)
            for i, (x, y) in enumerate(pts):
                cv2.circle(img, td.pt(x, y), 6, (0, 0, 220), -1)
                cv2.putText(img, str(i + 1), (td.pt(x, y)[0] + 8, td.pt(x, y)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1, cv2.LINE_AA)
            if len(pts) > 1:
                cv2.polylines(img, [np.array([td.pt(x, y) for x, y in pts], np.int32)], False, (0, 0, 220), 1, cv2.LINE_AA)
        cv2.imshow('draw the route', img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            say('not saved')
            return 0
        if k == ord('u') and pts:
            pts.pop()
            state['dirty'] = True
        if k == ord('s'):
            if state['cfg'] is None or state['probs']:
                say('cannot save: %s' % '; '.join(state['probs']))
                continue
            save(st, state['cfg'])
            say('Now run:  simulate --check')
            return 0


if __name__ == '__main__':
    sys.exit(main())
