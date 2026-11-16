"""Where to put a corner-mounted camera, and where to point it.

Give the camera's position; the tool searches the aim, scores how much of the
floor is usable, and draws two pictures: a floor map coloured by how many
pixels a tag cell gets, and a preview of what the camera should see. Hold the
preview next to the live image to aim the real camera.

    plan_mount --camera -0.1 -0.1 2.3            # floor size from arena.yaml
    plan_mount --camera 0 0 2.4 --floor 5.0 3.5 --tag-size 0.20
    plan_mount --camera 0 0 2.4 --look-at 2.2 1.6

Coordinates are arena metres; z is height above the floor.
"""
import argparse
import math
import sys

import numpy as np

from ..core.calib import GOOD_PX_PER_CELL, MIN_PX_PER_CELL, best_aim, coverage, score_coverage
from ..core.camera import FloorMap
from ..core.synth import Scene
from .common import cv2, say, settings, verdict


def floor_map_image(cov, bounds, cam_pos, look, path, px_per_m=140):
    x0, x1, y0, y1 = bounds
    pad = 0.6
    W, H = int((x1 - x0 + 2 * pad) * px_per_m), int((y1 - y0 + 2 * pad) * px_per_m)
    img = np.full((H, W, 3), 250, np.uint8)

    def pt(x, y):
        return int((x - x0 + pad) * px_per_m), int(H - (y - y0 + pad) * px_per_m)
    step = cov['xs'][1] - cov['xs'][0] if len(cov['xs']) > 1 else 0.25
    for j, y in enumerate(cov['ys']):
        for i, x in enumerate(cov['xs']):
            c = cov['px_per_cell'][j, i]
            if not cov['visible'][j, i]:
                col = (200, 200, 200)
            elif c >= GOOD_PX_PER_CELL:
                col = (150, 215, 150)
            elif c >= MIN_PX_PER_CELL:
                col = (130, 220, 245)
            else:
                col = (140, 140, 240)
            a, b = pt(x - step / 2, y + step / 2), pt(x + step / 2, y - step / 2)
            cv2.rectangle(img, a, b, col, -1)
            if cov['visible'][j, i]:
                cv2.putText(img, '%.1f' % c, (a[0] + 3, b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.rectangle(img, pt(x0, y1), pt(x1, y0), (60, 60, 60), 2)
    cp = pt(cam_pos[0], cam_pos[1])
    cv2.circle(img, cp, 8, (200, 60, 0), -1)
    cv2.arrowedLine(img, cp, pt(look[0], look[1]), (200, 60, 0), 2, cv2.LINE_AA, tipLength=0.05)
    cv2.putText(img, 'camera', (cp[0] + 10, cp[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 60, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'pixels per tag cell: green >= %.1f good, yellow >= %.1f marginal, red too small, grey not in view'
                % (GOOD_PX_PER_CELL, MIN_PX_PER_CELL), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog='plan_mount', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config-dir', dest='config_dir', default=None)
    ap.add_argument('--camera', nargs=3, type=float, required=True, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--look-at', dest='look', nargs=2, type=float, default=None, metavar=('X', 'Y'))
    ap.add_argument('--floor', nargs=2, type=float, default=None, metavar=('WIDTH', 'HEIGHT'))
    ap.add_argument('--tag-size', dest='tag_size', type=float, default=None)
    ap.add_argument('--tag-height', dest='tag_height', type=float, default=None)
    ap.add_argument('--out', default='/tmp/swarm_mount')
    ap.add_argument('--no-preview', action='store_true')
    args = ap.parse_args(argv)
    st = settings(args)
    cam = st.camera_model()
    tg = st.fleet.get('tags') or {}
    size = args.tag_size or float(tg.get('size', 0.15))
    height = args.tag_height if args.tag_height is not None else float(tg.get('height', 0.12))
    if args.floor:
        bounds = (0.0, args.floor[0], 0.0, args.floor[1])
    else:
        ar = st.arena()
        bounds = (ar.x0, ar.x1, ar.y0, ar.y1)
    hf, vf = cam.fov_deg()
    say('lens: %.0f x %.0f deg field of view at %dx%d' % (hf, vf, cam.size[0], cam.size[1]))
    pos = tuple(args.camera)
    if args.look:
        look = (args.look[0], args.look[1], 0.0)
        fm = FloorMap.from_mount(cam, pos, look)
    else:
        say('searching for the best aim ...')
        fm, look, _, _ = best_aim(cam, pos, bounds, size, height)
    cov = coverage(fm, bounds, size, height, step=0.25)
    sc = score_coverage(cov)
    m = fm.describe()
    say('')
    say('camera at (%.2f, %.2f), %.2f m above the floor' % pos)
    say('aim the CENTRE of the image at the floor point (%.2f, %.2f): that is %.0f deg below horizontal'
        % (look[0], look[1], m['pitch_down_deg']))
    say('')
    say('with %.0f mm tags at %.0f mm height:' % (1e3 * size, 1e3 * height))
    verdict(sc['visible_fraction'] > 0.97, '%.0f %% of the floor is in view' % (100 * sc['visible_fraction']))
    verdict(sc['usable_fraction'] > 0.97, '%.0f %% of the floor has a detectable tag (>= %.1f px per cell)'
            % (100 * sc['usable_fraction'], MIN_PX_PER_CELL))
    verdict(sc['good_fraction'] > 0.85, '%.0f %% is comfortable (>= %.1f px per cell); smallest %.1f'
            % (100 * sc['good_fraction'], GOOD_PX_PER_CELL, sc['min_px_per_cell']))
    say('  position noise from pixel noise alone: %.1f mm typical, %.1f mm at the worst spot'
        % (sc['median_sigma_mm'], sc['worst_sigma_mm']))
    if sc['usable_fraction'] <= 0.97:
        need = size * MIN_PX_PER_CELL / max(sc['min_px_per_cell'], 0.2) if sc['min_px_per_cell'] > 0 else size * 1.5
        say('  to fix: mount higher, move the far edge of the working area closer, or print bigger tags'
            ' (about %.0f mm would cover what is in view).' % (1e3 * math.ceil(need * 100) / 100))
    say('')
    say('floor map: %s' % floor_map_image(cov, bounds, pos, look, args.out + '_floor.png'))
    if not args.no_preview:
        sc2 = Scene(fm, seed=1)
        k = 0
        x0, x1, y0, y1 = bounds
        for x in np.arange(x0 + 0.5, x1 - 0.2, 1.0):
            for y in np.arange(y0 + 0.5, y1 - 0.2, 1.0):
                sc2.add_tag(k % 500, x, y, 0.0, z=height, size=size)
                k += 1
        img = cv2.cvtColor(sc2.render(ss=1), cv2.COLOR_GRAY2BGR)
        from ..core.viz import draw_floor_grid
        draw_floor_grid(img, fm, bounds)
        c = (int(cam.K[0, 2]), int(cam.K[1, 2]))
        cv2.drawMarker(img, c, (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.imwrite(args.out + '_preview.jpg', img)
        say('preview of the camera image (tags every metre, 0.5 m grid, red cross = image centre): %s_preview.jpg' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
