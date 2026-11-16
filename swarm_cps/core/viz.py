"""Top-down picture of the floor: walls, gate, zones, rock, the route and the
lanes, every rover with its slot and trail, and what the decider chose and
why. One function draws it for the simulator, for the live viewer and for the
figures in a report, so what is rehearsed looks like what is run.
"""
import math

import numpy as np

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None

ROVER_COLORS = [(40, 40, 40), (200, 110, 30), (60, 150, 60), (150, 60, 170), (30, 150, 200), (90, 90, 200)]


class TopDown:
    def __init__(self, arena, px_per_m=170, pad=0.25, panel=330):
        self.arena = arena
        self.k = float(px_per_m)
        self.pad = pad
        self.panel = panel
        self.w = int((arena.x1 - arena.x0 + 2 * pad) * self.k)
        self.h = int((arena.y1 - arena.y0 + 2 * pad) * self.k)
        self.trails = {}
        self.worked = None

    def pt(self, x, y):
        return (int(round((x - self.arena.x0 + self.pad) * self.k)),
                int(round(self.h - (y - self.arena.y0 + self.pad) * self.k)))

    def track(self, poses):
        """Record positions for the trails without drawing (call every step)."""
        for n, pz in poses.items():
            tr = self.trails.setdefault(n, [])
            if not tr or math.hypot(tr[-1][0] - pz[0], tr[-1][1] - pz[1]) > 0.03:
                tr.append((pz[0], pz[1]))

    def _text(self, img, s, org, scale=0.5, color=(30, 30, 30), bold=False):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)

    def draw(self, ctl=None, poses=None, metrics=None, title='', thought='', instruction='', extra=None):
        """poses: dict name -> (x, y, yaw). Returns a BGR image."""
        ar = self.arena
        img = np.full((self.h, self.w + self.panel, 3), 247, np.uint8)
        cv2.rectangle(img, self.pt(ar.x0, ar.y1), self.pt(ar.x1, ar.y0), (252, 252, 252), -1)
        # ground worked so far
        if metrics is not None and metrics.field is not None and metrics.worked.any():
            x0, x1, y0, y1 = metrics.field
            a, b = self.pt(x0, y1), self.pt(x1, y0)
            m = cv2.resize(metrics.worked[::-1].astype(np.uint8), (b[0] - a[0], b[1] - a[1]), interpolation=cv2.INTER_NEAREST)
            sub = img[a[1]:b[1], a[0]:b[0]]
            sub[m > 0] = (225, 240, 225)
        for z in ar.zones:
            a, b = self.pt(z['x0'], z['y1']), self.pt(z['x1'], z['y0'])
            ov = img.copy()
            cv2.rectangle(ov, a, b, (170, 200, 255), -1)
            img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
            cv2.rectangle(img, a, b, (60, 120, 230), 1)
            self._text(img, z['name'], (a[0] + 6, a[1] + 18), 0.5, (40, 80, 190))
        cv2.rectangle(img, self.pt(ar.x0, ar.y1), self.pt(ar.x1, ar.y0), (120, 120, 120), 2)
        for w in ar.walls:
            cv2.line(img, self.pt(*w['a']), self.pt(*w['b']), (50, 50, 50), 5)
        for o in ar.obstacles:
            cv2.circle(img, self.pt(o['x'], o['y']), max(3, int(o['r'] * self.k)), (95, 95, 95), -1)
            self._text(img, o['name'], (self.pt(o['x'], o['y'])[0] + 8, self.pt(o['x'], o['y'])[1] - 10), 0.45, (80, 80, 80))
        if ctl is not None:
            r = ctl.route
            i0, i1 = r.index(0.0), r.index(r.length)
            pts = np.array([self.pt(x, y) for x, y in zip(r.x[i0:i1:4], r.y[i0:i1:4])], np.int32)
            cv2.polylines(img, [pts], False, (190, 190, 190), 1, cv2.LINE_AA)
            for lg in r.legs:                                 # working legs drawn heavier
                if lg['work']:
                    p2, _ = r.lane_samples(0.0, lg['s0'], lg['s1'], 0.08)
                    cv2.polylines(img, [np.array([self.pt(x, y) for x, y in p2], np.int32)], False, (150, 190, 150), 2, cv2.LINE_AA)
            for sr in getattr(ctl, 'solo', {}).values():
                j0, j1 = sr.index(0.0), sr.index(sr.length)
                sp = np.array([self.pt(x, y) for x, y in zip(sr.x[j0:j1:4], sr.y[j0:j1:4])], np.int32)
                cv2.polylines(img, [sp], False, (200, 170, 200), 1, cv2.LINE_AA)
            if ctl.leader in ctl.s:
                for name, (sx, sy, _) in ctl.slot_positions().items():
                    cv2.circle(img, self.pt(sx, sy), int(0.05 * self.k), (150, 150, 150), 1, cv2.LINE_AA)
        if poses:
            names = list(ctl.names) if ctl is not None else sorted(poses)
            for i, n in enumerate(names):
                if n not in poses:
                    continue
                x, y, yaw = poses[n][:3]
                col = ROVER_COLORS[i % len(ROVER_COLORS)]
                self.track({n: (x, y)})
                tr = self.trails[n]
                if len(tr) > 1:
                    cv2.polylines(img, [np.array([self.pt(a, b) for a, b in tr], np.int32)], False, col, 1, cv2.LINE_AA)
                L, Wd = 0.213, 0.159                           # MentorPi A1 footprint
                c, s = math.cos(yaw), math.sin(yaw)
                box = [(x + a * c - b * s, y + a * s + b * c) for a, b in ((L / 2, Wd / 2), (L / 2, -Wd / 2), (-L / 2, -Wd / 2), (-L / 2, Wd / 2))]
                cv2.fillPoly(img, [np.array([self.pt(a, b) for a, b in box], np.int32)], col, cv2.LINE_AA)
                cv2.line(img, self.pt(x, y), self.pt(x + 0.17 * c, y + 0.17 * s), (255, 255, 255), 2, cv2.LINE_AA)
                stopped = ctl is not None and n in ctl.stop_cause
                self._text(img, n + (' (stopped)' if stopped else ''), (self.pt(x, y)[0] + 12, self.pt(x, y)[1] - 12), 0.42,
                           (0, 0, 200) if stopped else col)
        # compass and scale
        ox, oy = self.pt(ar.x0, ar.y0)
        cv2.arrowedLine(img, (ox + 8, oy - 8), (ox + 8 + int(0.5 * self.k), oy - 8), (140, 140, 140), 1, cv2.LINE_AA, tipLength=0.15)
        self._text(img, '%s (+x), 0.5 m' % ar.plus_x, (ox + 12, oy - 14), 0.4, (120, 120, 120))
        self._text(img, '%s (+y) up' % ar.plus_y, (ox + 12, oy - 30), 0.4, (120, 120, 120))
        # side panel
        x0 = self.w + 12
        y = 28
        if title:
            self._text(img, title, (x0, y), 0.62, (20, 20, 20), True)
            y += 30
        lines = []
        if instruction:
            lines += ['Instruction:'] + _wrap('"%s"' % instruction, 34) + ['']
        if ctl is not None:
            st = ctl.status
            lines += ['Phase: %s' % st.get('phase', ''), 'Formation: %s' % ctl.formation,
                      'Speed: %.2f m/s' % ctl.speed, 'Offset: %+.2f m (%s +)' % (ctl.offset_applied, ar.offset_words(ctl.route)[0])]
            if ctl.wanted is not None:
                lines += ['Held by safety filter: %s' % ctl.wanted['name']]
            if st.get('fleet_stop'):
                lines += ['FLEET STOP: %s' % st['fleet_stop']]
            lines += ['']
        if thought:
            lines += ['Decider:'] + _wrap(thought, 34) + ['']
        if metrics is not None:
            res = metrics.result()
            lines += ['Time: %.0f s' % res['time_s'], 'Collisions: %d' % res['collisions'],
                      'Ground worked: %.0f %%' % res['field_worked_pct']]
            for k, v in res['zone_machine_seconds'].items():
                lines += ['Machine-seconds in', '  %s: %.0f' % (k, v)]
        for ln in (extra or []):
            lines.append(ln)
        for ln in lines:
            if y > self.h - 8:
                break
            self._text(img, ln, (x0, y), 0.46, (0, 0, 190) if ln.startswith('FLEET') else (40, 40, 40))
            y += 20
        return img


def draw_floor_grid(img, fm, bounds, step=0.5, color=(255, 160, 0)):
    """Draw the arena's 0.5 m grid on the camera image as the calibration
    says it lies. If the lines sit on the floor where your tape marks are,
    the calibration is right; this is the quickest check there is."""
    x0, x1, y0, y1 = bounds

    def line(a, b):
        n = 40
        pts = np.array([[a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, 0.0] for k in range(n + 1)])
        px, front = fm.world_to_pixel(pts)
        ok = front & fm.in_image(px)
        for k in range(n):
            if ok[k] and ok[k + 1]:
                cv2.line(img, (int(px[k][0]), int(px[k][1])), (int(px[k + 1][0]), int(px[k + 1][1])), color, 1, cv2.LINE_AA)
    x = x0
    while x <= x1 + 1e-9:
        line((x, y0), (x, y1))
        x += step
    y = y0
    while y <= y1 + 1e-9:
        line((x0, y), (x1, y))
        y += step
    px, front = fm.world_to_pixel([[x0, y0, 0.0]])
    if front[0] and fm.in_image(px)[0]:
        cv2.putText(img, '(%.1f, %.1f)' % (x0, y0), (int(px[0][0]) + 6, int(px[0][1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _wrap(text, width):
    out, line = [], ''
    for word in str(text).split():
        if len(line) + len(word) + 1 > width and line:
            out.append(line)
            line = word
        else:
            line = (line + ' ' + word).strip()
    if line:
        out.append(line)
    return out


def write_gif(frames, path, fps=10):
    """Save BGR frames as a GIF if Pillow is available. Returns True on success."""
    try:
        from PIL import Image
    except Exception:
        return False
    ims = [Image.fromarray(f[:, :, ::-1]).quantize(colors=64, method=0) for f in frames]
    if not ims:
        return False
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=False)
    return True
