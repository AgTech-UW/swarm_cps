"""The floor: bounds, walls (and the gate between them), free-standing
obstacles, named zones, and the mission route. Loaded from arena.yaml.

Coordinates are metres in the arena frame set by the floor calibration:
+x is "east" and +y is "north" in everything the operator and the supervisor
read. Those are just names for the two floor axes; they need not match a
compass.
"""
import math

import numpy as np

from .path import Route


def _seg_point_dist(p, a, b):
    ax, ay = a
    bx, by = b
    ux, uy = bx - ax, by - ay
    L2 = ux * ux + uy * uy
    if L2 < 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * ux + (p[1] - ay) * uy) / L2))
    return math.hypot(p[0] - ax - t * ux, p[1] - ay - t * uy)


def _seg_seg_dist(p1, p2, q1, q2):
    """Distance between segments p1-p2 and q1-q2 (0 if they cross)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return 0.0
    return min(_seg_point_dist(p1, q1, q2), _seg_point_dist(p2, q1, q2),
               _seg_point_dist(q1, p1, p2), _seg_point_dist(q2, p1, p2))


class Arena:
    def __init__(self, cfg):
        self.cfg = cfg
        b = cfg['bounds']
        self.x0, self.x1 = float(b['x'][0]), float(b['x'][1])
        self.y0, self.y1 = float(b['y'][0]), float(b['y'][1])
        self.edge_margin = float(cfg.get('edge_margin', 0.20))
        comp = cfg.get('compass', {})
        self.plus_x = str(comp.get('plus_x', 'east'))
        self.plus_y = str(comp.get('plus_y', 'north'))
        self.walls = []
        for w in cfg.get('walls') or []:
            self.walls.append(dict(name=str(w.get('name', 'wall')), a=(float(w['from'][0]), float(w['from'][1])),
                                   b=(float(w['to'][0]), float(w['to'][1]))))
        self.obstacles = []
        for o in cfg.get('obstacles') or []:
            self.obstacles.append(dict(name=str(o.get('name', 'obstacle')), x=float(o['x']), y=float(o['y']),
                                       r=float(o['r'])))
        self.zones = []
        for z in cfg.get('zones') or []:
            self.zones.append(dict(name=str(z['name']), x0=float(min(z['x'])), x1=float(max(z['x'])),
                                   y0=float(min(z['y'])), y1=float(max(z['y'])),
                                   kind=str(z.get('kind', 'area'))))
        m = cfg['mission']
        self.turn_radius = float(m.get('turn_radius', 0.55))
        self.swath = float(m.get('swath', 0.30))
        st = m['start']
        self.start = (float(st[0]), float(st[1]), math.radians(float(st[2])))
        self.legs = []
        for lg in m['legs']:
            t = lg['to']
            self.legs.append(dict(name=str(lg['name']), to=(float(t[0]), float(t[1]), math.radians(float(t[2]))),
                                  work=bool(lg.get('work', False))))
        self.lead_in = float(m.get('lead_in', 3.0))
        self.solo = []                                        # rovers that drive a route of their own
        for sd in m.get('solo') or []:
            st2 = sd['start']
            self.solo.append(dict(robot=str(sd.get('robot', 'last')),
                                  start=(float(st2[0]), float(st2[1]), math.radians(float(st2[2]))),
                                  legs=[dict(name=str(lg.get('name', 'solo')), work=False,
                                             to=(float(lg['to'][0]), float(lg['to'][1]), math.radians(float(lg['to'][2]))))
                                        for lg in sd['legs']]))
        self.offset_ramp = float(m.get('offset_ramp', 0.6))

    @classmethod
    def load(cls, path):
        import yaml
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def make_solo_route(self, sd):
        return Route.from_poses(sd['start'], sd['legs'], self.turn_radius, ds=0.02, lead_in=0.6, run_out=1.0)

    def make_route(self):
        return Route.from_poses(self.start, self.legs, self.turn_radius, ds=0.02, lead_in=self.lead_in,
                                run_out=1.0, ramp=self.offset_ramp)

    # ---- geometry ----
    def inside(self, x, y, margin=None):
        m = self.edge_margin if margin is None else margin
        return (self.x0 + m <= x <= self.x1 - m) and (self.y0 + m <= y <= self.y1 - m)

    def wall_distance(self, x, y):
        """Distance from a point to the nearest wall (inf if there are none)."""
        d = float('inf')
        for w in self.walls:
            d = min(d, _seg_point_dist((x, y), w['a'], w['b']))
        return d

    def nearest_wall(self, x, y):
        best = (float('inf'), None)
        for w in self.walls:
            d = _seg_point_dist((x, y), w['a'], w['b'])
            if d < best[0]:
                best = (d, w)
        return best

    def segment_wall_distance(self, p, q):
        d = float('inf')
        for w in self.walls:
            d = min(d, _seg_seg_dist(p, q, w['a'], w['b']))
        return d

    def points_free(self, pts, clearance, margin=None):
        """Are all points (N, 2) inside the bounds and at least `clearance`
        from every wall? Returns (ok, reason)."""
        m = self.edge_margin if margin is None else margin
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) == 0:
            return True, ''
        if (pts[:, 0] < self.x0 + m - 1e-9).any() or (pts[:, 0] > self.x1 - m + 1e-9).any() or \
           (pts[:, 1] < self.y0 + m - 1e-9).any() or (pts[:, 1] > self.y1 - m + 1e-9).any():
            return False, 'leaves the floor area'
        for w in self.walls:
            a, b = np.array(w['a']), np.array(w['b'])
            u = b - a
            L2 = float(u @ u)
            t = np.clip(((pts - a) @ u) / max(L2, 1e-12), 0.0, 1.0)
            d = np.linalg.norm(pts - (a + t[:, None] * u), axis=1)
            if (d < clearance).any():
                return False, 'too close to %s' % w['name']
        return True, ''

    def zones_at(self, x, y):
        return [z['name'] for z in self.zones if z['x0'] <= x <= z['x1'] and z['y0'] <= y <= z['y1']]

    def compass(self, heading):
        """Heading (rad) -> compass word in the arena's own naming."""
        opp = {'east': 'west', 'west': 'east', 'north': 'south', 'south': 'north'}
        px, py = self.plus_x, self.plus_y
        names = [px, '%s-%s' % (py, px), py, '%s-%s' % (py, opp.get(px, 'minus-x')), opp.get(px, 'minus-x'),
                 '%s-%s' % (opp.get(py, 'minus-y'), opp.get(px, 'minus-x')), opp.get(py, 'minus-y'),
                 '%s-%s' % (opp.get(py, 'minus-y'), px)]
        d = (math.degrees(heading) + 360.0) % 360.0
        return names[int((d + 22.5) // 45) % 8]

    def offset_words(self, route):
        """(word for + offset, word for - offset), e.g. ('north', 'south')."""
        opp = {'east': 'west', 'west': 'east', 'north': 'south', 'south': 'north'}
        plus = self.plus_y if abs(route.axis[1]) > 0.5 else self.plus_x
        return plus, opp.get(plus, 'the other way')

    def describe_position(self, x, y):
        """Rough words for where something is, for the supervisor."""
        fx = (x - self.x0) / max(self.x1 - self.x0, 1e-9)
        fy = (y - self.y0) / max(self.y1 - self.y0, 1e-9)
        opp = {'east': 'west', 'west': 'east', 'north': 'south', 'south': 'north'}
        ew = self.plus_x if fx > 0.6 else (opp.get(self.plus_x, 'low-x') if fx < 0.4 else 'middle')
        ns = self.plus_y if fy > 0.6 else (opp.get(self.plus_y, 'low-y') if fy < 0.4 else 'middle')
        if ew == 'middle' and ns == 'middle':
            return 'middle of the floor'
        if ew == 'middle':
            return '%s side' % ns
        if ns == 'middle':
            return '%s side' % ew
        return '%s-%s corner' % (ns, ew)

    def validate(self, route, body_radius, min_turn_radius):
        """Problems with the layout that would stop a run, as a list of strings."""
        out = []
        pts, _ = route.lane_samples(0.0, 0.0, route.length, 0.05)
        ok, why = self.points_free(pts, body_radius + 0.05)
        if not ok:
            out.append('the leader\'s route %s (with %.2f m clearance)' % (why, body_radius + 0.05))
        for sd in self.solo:
            r2 = self.make_solo_route(sd)
            pts2, _ = r2.lane_samples(0.0, 0.0, r2.length, 0.05)
            ok2, why2 = self.points_free(pts2, body_radius + 0.05)
            if not ok2:
                out.append('the solo route of %s %s' % (sd['robot'], why2))
        if self.turn_radius < min_turn_radius:
            out.append('mission.turn_radius %.2f m is tighter than the rover can turn (%.2f m)'
                       % (self.turn_radius, min_turn_radius))
        for o in self.obstacles:
            if not self.inside(o['x'], o['y'], 0.0):
                out.append('obstacle %s is outside the bounds' % o['name'])
        if not any(lg['work'] for lg in self.legs):
            out.append('no leg is marked work: true, so the offset has nothing to act on')
        return out
