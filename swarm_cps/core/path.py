"""Routes for turn-limited rovers: Dubins connectors between poses, sampled to
a dense polyline that the vector-field guidance follows.

A Route is the leader's nominal line. It runs from s = 0 to s = length, with a
straight lead-in before 0 (so followers' slots behind the leader exist at the
start) and a straight run-out after the end. Each sample carries its heading,
curvature, leg name and an offset weight w in [0, 1]: the operator-facing
"offset" moves the route sideways by offset * w, so w = 0 keeps the yard and
the gate where they are and w = 1 lets the working passes (and the turn
between them) shift as one piece.
"""
import math

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _mod2pi(a):
    return a % (2 * math.pi)


def dubins_shortest(q0, q1, R):
    """Shortest Dubins path from pose q0 to q1 = (x, y, heading) with turning
    radius R. Returns [(kind, length_m)], kind in 'L', 'S', 'R'."""
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    D = math.hypot(dx, dy) / R
    th = math.atan2(dy, dx)
    a, b = _mod2pi(q0[2] - th), _mod2pi(q1[2] - th)
    sa, ca, sb, cb = math.sin(a), math.cos(a), math.sin(b), math.cos(b)
    cab = math.cos(a - b)
    cands = []
    # LSL
    p2 = 2 + D * D - 2 * cab + 2 * D * (sa - sb)
    if p2 >= 0:
        tmp = math.atan2(cb - ca, D + sa - sb)
        cands.append(('LSL', _mod2pi(-a + tmp), math.sqrt(p2), _mod2pi(b - tmp)))
    # RSR
    p2 = 2 + D * D - 2 * cab + 2 * D * (sb - sa)
    if p2 >= 0:
        tmp = math.atan2(ca - cb, D - sa + sb)
        cands.append(('RSR', _mod2pi(a - tmp), math.sqrt(p2), _mod2pi(-b + tmp)))
    # LSR
    p2 = -2 + D * D + 2 * cab + 2 * D * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, D + sa + sb) - math.atan2(-2.0, p)
        cands.append(('LSR', _mod2pi(-a + tmp), p, _mod2pi(-_mod2pi(b) + tmp)))
    # RSL
    p2 = D * D - 2 + 2 * cab - 2 * D * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, D - sa - sb) - math.atan2(2.0, p)
        cands.append(('RSL', _mod2pi(a - tmp), p, _mod2pi(b - tmp)))
    # RLR
    tmp = (6 - D * D + 2 * cab + 2 * D * (sa - sb)) / 8
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(a - math.atan2(ca - cb, D - sa + sb) + p / 2)
        cands.append(('RLR', t, p, _mod2pi(a - b - t + p)))
    # LRL
    tmp = (6 - D * D + 2 * cab + 2 * D * (sb - sa)) / 8
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(-a - math.atan2(ca - cb, D + sa - sb) + p / 2)
        cands.append(('LRL', t, p, _mod2pi(_mod2pi(b) - a - t + p)))
    best = None
    for word, t, p, q in cands:
        # a full extra circle is never the shortest way to go nowhere
        segs = [(word[0], t * R), (word[1], p * R), (word[2], q * R)]
        end = _integrate(q0, segs, R)
        err = math.hypot(end[0] - q1[0], end[1] - q1[1]) + abs(wrap(end[2] - q1[2])) * R
        if err > 1e-6 * max(1.0, D * R) + 1e-6:
            continue                                          # numerically invalid word
        L = (t + p + q) * R
        if best is None or L < best[0]:
            best = (L, segs)
    if best is None:
        raise ValueError('no Dubins path from %r to %r with radius %.3f' % (q0, q1, R))
    return [(k, l) for k, l in best[1] if l > 1e-9]


def _advance(q, kind, length, R):
    x, y, th = q
    if kind == 'S':
        return (x + length * math.cos(th), y + length * math.sin(th), th)
    sg = 1.0 if kind == 'L' else -1.0
    th2 = th + sg * length / R
    return (x + sg * R * (math.sin(th2) - math.sin(th)), y - sg * R * (math.cos(th2) - math.cos(th)), th2)


def _integrate(q0, segs, R):
    q = q0
    for kind, length in segs:
        q = _advance(q, kind, length, R)
    return q


def sample_segments(q0, segs, R, ds):
    """Points (x, y, heading, curvature) every ds along a Dubins path,
    excluding the start point, including the end point."""
    out = []
    q = q0
    carry = 0.0
    for kind, length in segs:
        k = 0.0 if kind == 'S' else (1.0 / R if kind == 'L' else -1.0 / R)
        d = ds - carry
        while d <= length + 1e-12:
            p = _advance(q, kind, d, R)
            out.append((p[0], p[1], p[2], k))
            d += ds
        carry = length - (d - ds)
        q = _advance(q, kind, length, R)
    return out, q


class Route:
    def __init__(self, x, y, s, leg, legs, weight, length, ds, offset_axis):
        self.x0 = np.asarray(x, float)              # nominal samples
        self.y0 = np.asarray(y, float)
        self.s = np.asarray(s, float)               # arc length of each sample (negative in the lead-in)
        self.leg = np.asarray(leg, int)             # index into self.legs per sample
        self.legs = legs                            # [dict(name, work, s0, s1)]
        self.w = np.asarray(weight, float)
        self.length = float(length)
        self.ds = float(ds)
        self.axis = np.asarray(offset_axis, float)  # unit vector that + offset moves the route along
        self.span, self.ramp_min, self.turn_radius = None, 0.6, 0.5
        self.offset = None
        self.set_offset(0.0)

    # ---- construction ----
    @classmethod
    def from_poses(cls, start, legs, turn_radius, ds=0.02, lead_in=3.0, run_out=1.0, ramp=0.6):
        """start: (x, y, heading_rad). legs: [dict(name, to=(x, y, heading_rad),
        work=bool)]. Consecutive poses are joined by the shortest Dubins path
        of radius turn_radius (a straight line when they are in line)."""
        pts = [(start[0], start[1], start[2], 0.0)]
        leg_idx = [0]
        info = []
        q = tuple(start)
        s_acc = 0.0
        for i, lg in enumerate(legs):
            to = tuple(lg['to'])
            segs = dubins_shortest(q, to, turn_radius)
            L = sum(l for _, l in segs)
            smp, q_end = sample_segments(q, segs, turn_radius, ds)
            # sampling carries across segments, so close the leg on its exact end pose
            if not smp or math.hypot(smp[-1][0] - to[0], smp[-1][1] - to[1]) > 1e-6:
                smp.append((to[0], to[1], to[2], 0.0))
            pts.extend(smp)
            leg_idx.extend([i] * len(smp))
            info.append(dict(name=str(lg.get('name', 'leg%d' % i)), work=bool(lg.get('work', False)),
                             s0=s_acc, s1=s_acc + L, turn=max((abs(l / turn_radius) for k, l in segs if k != 'S'),
                                                              default=0.0)))
            s_acc += L
            q = to
        P = np.array(pts)
        raw_leg = np.array(leg_idx)
        # arc length along the sampled polyline, then resample uniformly so that
        # arc length <-> sample index is a plain division everywhere
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        keep = np.concatenate([[True], seg > 1e-9])
        P, raw_leg = P[keep], raw_leg[keep]
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        s_raw = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(s_raw[-1])
        scale = s_acc / length if length > 0 else 1.0
        for d in info:                               # keep leg boundaries consistent with s
            d['s0'] /= scale
            d['s1'] /= scale
        n = max(2, int(round(length / ds)))
        ds = length / n
        s = np.linspace(0.0, length, n + 1)
        px = np.interp(s, s_raw, P[:, 0])
        py = np.interp(s, s_raw, P[:, 1])
        leg_u = raw_leg[np.clip(np.searchsorted(s_raw, s, side='left'), 0, len(raw_leg) - 1)]
        # straight lead-in and run-out
        n_in, n_out = int(round(lead_in / ds)), int(round(run_out / ds))
        th0, th1 = start[2], q[2]
        xin = [px[0] - (n_in - k) * ds * math.cos(th0) for k in range(n_in)]
        yin = [py[0] - (n_in - k) * ds * math.sin(th0) for k in range(n_in)]
        xout = [px[-1] + (k + 1) * ds * math.cos(th1) for k in range(n_out)]
        yout = [py[-1] + (k + 1) * ds * math.sin(th1) for k in range(n_out)]
        x = np.concatenate([xin, px, xout])
        y = np.concatenate([yin, py, yout])
        sfull = np.concatenate([[-(n_in - k) * ds for k in range(n_in)], s, [length + (k + 1) * ds for k in range(n_out)]])
        leg = np.concatenate([[0] * n_in, leg_u, [len(legs) - 1] * n_out])
        # the offset applies from the start of the first working leg to the end of the last
        work = [d for d in info if d['work']]
        axis = np.array([0.0, 1.0])
        span = None
        if work:
            a, b = work[0]['s0'], work[-1]['s1']
            span = (a, b, abs(b - length) < 1e-6)
            i0 = int(np.searchsorted(sfull, a)) + 2
            hd = math.atan2(y[min(i0 + 5, len(y) - 1)] - y[i0], x[min(i0 + 5, len(x) - 1)] - x[i0])
            # passes running east-west shift north (+) / south (-); north-south ones shift east / west
            axis = np.array([0.0, 1.0]) if abs(math.cos(hd)) >= abs(math.sin(hd)) else np.array([1.0, 0.0])
        r = cls(x, y, sfull, leg, info, np.zeros(len(x)), length, ds, axis)
        r.span, r.ramp_min, r.turn_radius = span, float(ramp), float(turn_radius)
        r.offset = None
        r.set_offset(0.0)
        return r

    # ---- offset ----
    def set_offset(self, offset):
        offset = float(offset)
        if self.offset is not None and abs(offset - self.offset) < 1e-4:
            return
        self.offset = offset
        span = getattr(self, 'span', None)
        if span is not None:
            # Ramp the shift in over a length the rover can actually steer:
            # a smoothstep of height h over length L has peak curvature 6 h / L^2.
            a, b, ends = span
            L = max(self.ramp_min, math.sqrt(6.0 * abs(offset) * self.turn_radius))
            L = min(L, max(0.2, 0.5 * (b - a)))
            up = np.clip((self.s - a) / L, 0.0, 1.0)
            dn = np.ones(len(self.s)) if ends else np.clip((b - self.s) / L, 0.0, 1.0)
            w = np.minimum(up, dn)
            self.w = w * w * (3 - 2 * w)
        self.x = self.x0 + self.w * offset * self.axis[0]
        self.y = self.y0 + self.w * offset * self.axis[1]
        dx, dy = np.gradient(self.x, self.s), np.gradient(self.y, self.s)
        self.th = np.unwrap(np.arctan2(dy, dx))
        k = np.gradient(self.th, self.s)
        # the samples are 2 cm apart; smooth the differentiated curvature a little
        kk = np.convolve(k, np.ones(5) / 5, mode='same')
        self.kappa = kk

    # ---- queries ----
    def index(self, s):
        return int(np.clip(round((s - self.s[0]) / self.ds), 0, len(self.s) - 1))

    def at(self, s):
        """(x, y, heading, curvature) at arc length s (clamped to the route)."""
        f = (s - self.s[0]) / self.ds
        i = int(np.clip(math.floor(f), 0, len(self.s) - 2))
        a = float(np.clip(f - i, 0.0, 1.0))
        return (float(self.x[i] + a * (self.x[i + 1] - self.x[i])), float(self.y[i] + a * (self.y[i + 1] - self.y[i])),
                float(self.th[i] + a * (self.th[i + 1] - self.th[i])), float(self.kappa[i] + a * (self.kappa[i + 1] - self.kappa[i])))

    def lane_point(self, s, lat):
        """Point `lat` metres to the left of the route at arc length s."""
        x, y, th, k = self.at(s)
        return x - lat * math.sin(th), y + lat * math.cos(th), th, k

    def leg_at(self, s):
        return self.legs[int(self.leg[self.index(min(max(s, 0.0), self.length))])]

    def project(self, px, py, s_hint=None, back=0.4, ahead=1.0):
        """Nearest point of the route to (px, py), searched in a window around
        s_hint (the whole route if None). Returns (s, e) with e the signed
        cross-track distance, + when the point is left of the route."""
        if s_hint is None:
            i0, i1 = 0, len(self.s)
        else:
            i0 = max(0, self.index(s_hint - back))
            i1 = min(len(self.s), self.index(s_hint + ahead) + 1)
            if i1 - i0 < 2:
                i0 = max(0, i1 - 2)
        d2 = (self.x[i0:i1] - px) ** 2 + (self.y[i0:i1] - py) ** 2
        i = i0 + int(np.argmin(d2))
        # refine along the two neighbouring segments
        best = (float('inf'), self.s[i], 0.0)
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self.s):
                continue
            ax, ay, bx, by = self.x[j], self.y[j], self.x[j + 1], self.y[j + 1]
            ux, uy = bx - ax, by - ay
            L2 = ux * ux + uy * uy
            if L2 < 1e-12:
                continue
            t = float(np.clip(((px - ax) * ux + (py - ay) * uy) / L2, 0.0, 1.0))
            cx, cy = ax + t * ux, ay + t * uy
            d = math.hypot(px - cx, py - cy)
            if d < best[0]:
                e = (ux * (py - ay) - uy * (px - ax)) / math.sqrt(L2)
                best = (d, float(self.s[j] + t * (self.s[j + 1] - self.s[j])), float(e))
        return best[1], best[2]

    def max_curvature(self, s0, s1):
        i0, i1 = self.index(s0), self.index(s1) + 1
        k = self.kappa[i0:i1]
        return (float(k.max()), float(k.min())) if len(k) else (0.0, 0.0)

    def lane_samples(self, lat, s0, s1, step=0.05):
        """Points along the lane `lat` metres left of the route between s0 and s1."""
        n = max(2, int(abs(s1 - s0) / step) + 1)
        ss = np.linspace(s0, s1, n)
        idx = np.clip(np.round((ss - self.s[0]) / self.ds).astype(int), 0, len(self.s) - 1)
        return np.stack([self.x[idx] - lat * np.sin(self.th[idx]), self.y[idx] + lat * np.cos(self.th[idx])], 1), ss
