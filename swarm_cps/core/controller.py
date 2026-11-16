"""The crew controller: one leader on a route, followers in slots, boid
separation, an obstacle reflex, and the hard stops underneath.

It is the Part I / Foreman controller written directly in metres and seconds
for small Ackermann rovers:

  leader      vector-field path following (Nelson et al.) along the route
  followers   each drives a line through its slot with the same vector field
              and holds its place along the line with its speed (Foreman Eq. 1)
  separation  Reynolds' rule with the inverse-distance profile, range P
  reflex      a machine about to run into a free-standing obstacle bends its
              course to the edge of the obstacle's cone, on a side it picks once
              and keeps, and the flock rules may not turn it back (Foreman Eq. 2)
  order       a total order over machines decides who brakes for whom
  limits      speed and yaw rate are always scaled together, so a slowed
              command never asks for a tighter turn (Faux-GPS Eq. 12-13)

Under those sit stops that do not negotiate: stale poses, leaving the floor,
a machine in the way, a wall. This layer decides nothing about WHAT the crew
does. Formation, speed and offset come in through request(), and pass a fit
check and a safety filter that the decider cannot override.

The controller never integrates motion. Every call to step() takes measured
poses and returns commands; the rovers (or the simulator) supply the next
poses.
"""
import math

import numpy as np

from . import formation as F
from .path import wrap


class Params:
    """Tunable numbers, all SI. Defaults suit a MentorPi A1 (213 x 159 mm)."""
    body_radius = 0.14          # m, disc that contains the rover
    min_turn_radius = 0.35      # m, measured chassis limit (assumed until measured)
    v_max = 0.22                # m/s, final cap on any command
    w_max = 1.0                 # rad/s, final cap
    speed_range = (0.08, 0.18)  # m/s, what a decider may ask of the leader
    speed_default = 0.13
    spacing = 0.45              # m, slot spacing (delta)
    protected = 0.38            # m, separation range (P); keep spacing > protected
    chi_inf_leader = math.radians(70.0)
    chi_inf_follower = math.radians(60.0)
    k_cross = None              # 1/m; None -> 2.5 / turn radius (the paper's 0.5 at R = 5)
    k_along = 0.8               # 1/s, follower speed per metre of along-track error
    k_heading = 2.5             # 1/s, yaw rate per radian of course error
    k_sep = 0.07                # m^2/s, separation strength
    brake_start = 0.50          # m, centre distance at which a lower-ranked machine starts to brake
    brake_stop = 0.34           # m, centre distance at which it has stopped
    stop_gap = 0.06             # m, body-to-body gap that stops any machine driving at another
    reflex_look = 1.0           # m, how far ahead the reflex reacts
    reflex_margin = 0.12        # m, clearance the reflex keeps
    stale_timeout = 0.5         # s, oldest pose the controller will act on
    leader_wait = (0.30, 0.90, 0.30)   # m lag where the leader starts to slow, m where it is slowest, floor
    slot_rate = 0.12            # m/s, how fast a slot moves to its new place on a formation change
    offset_rate = 0.05          # m/s, how fast the pass line moves sideways
    max_offset = 1.0            # m
    fit_reach = 0.7             # m, how far ahead a formation must fit
    fit_margin = 0.08           # m, spare clearance from walls for a formation to count as fitting
    sensor_range = 1.5          # m, hazards closer than this are reported to the decider
    period = 20.0               # s, longest time between decision points
    room_for = 4.0              # s, a wider formation must have fitted this long before it is offered
    goal_tolerance = 0.04       # m
    geofence_slack = 0.05       # m, a rover this far outside the bounds stops the fleet

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(Params, k):
                raise ValueError('unknown control parameter %r' % k)
            setattr(self, k, tuple(v) if isinstance(v, list) else v)


def limit_command(v, w, p):
    """Final chassis limits. Speed is capped first and the yaw rate is scaled
    by the same factor, then bounded by what the turning radius allows at that
    speed. A non-positive speed gives (0, 0): an Ackermann rover cannot turn
    in place, and these controllers never reverse."""
    if not (math.isfinite(v) and math.isfinite(w)) or v <= 1e-4:
        return 0.0, 0.0
    v2 = min(v, p.v_max)
    w2 = w * v2 / v
    lim = min(p.w_max, v2 / p.min_turn_radius)
    return v2, max(-lim, min(lim, w2))


class RobotState:
    __slots__ = ('x', 'y', 'yaw', 'v', 'age')

    def __init__(self, x, y, yaw, v=0.0, age=0.0):
        self.x, self.y, self.yaw, self.v, self.age = float(x), float(y), float(yaw), float(v), float(age)


class SwarmController:
    def __init__(self, arena, names, params=None, leader=None):
        """names: rover names in rank order, leader first (or name it with
        `leader`). Rank decides who yields: the leader to nobody, follower k
        to everyone before it."""
        self.arena = arena
        self.p = params or Params()
        names = list(names)
        if leader is not None:
            names.remove(leader)
            names.insert(0, leader)
        if len(names) < 1:
            raise ValueError('need at least a leader')
        self.names = names
        self.leader = names[0]
        self.followers = names[1:]
        self.n = len(self.followers)
        self.route = arena.make_route()
        if self.p.k_cross is None:
            self.k_cross = 2.5 / arena.turn_radius
        else:
            self.k_cross = float(self.p.k_cross)
        self._limits_cache = {}
        self.reset()

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        p = self.p
        self.armed = False
        self.done = False
        self.latched = None                 # a stop that needs reset(): 'geofence:rover2', ...
        self.t = 0.0
        self.t_armed = 0.0
        self.route.set_offset(0.0)
        self.offset_target = 0.0
        self.offset_applied = 0.0
        self.speed = p.speed_default
        self.formation = 'column'
        self.slots_target = F.preset_slots('column', self.n, p.spacing)
        self.assign = list(range(self.n))                    # follower index -> slot index
        self.slots_now = list(self.slots_target)             # per follower, morphing toward its slot
        self.s = {}                                          # progress along the route per rover
        self.s_leader = 0.0
        self.side = {}                                       # (rover, obstacle index) -> +1 / -1
        self.wanted = None                                   # a decision the safety filter is holding
        self.log = []                                        # decisions, holds, releases
        self.stop_cause = {}
        self.fleet_stop = None
        self.status = {}
        self._trig = dict(last=-1e9, phase=None, misfit=0, since={}, offered=set(), seen=set(), checked=-1e9,
                          started=False)
        self._rejected = None
        self._leader_cmd_v = 0.0
        self._lag = 0.0
        self._fit_checked = -1e9
        self._misfit_since = None

    def arm(self):
        if self.latched:
            return False, 'cannot arm: %s (reset first)' % self.latched
        self.armed = True
        self.t_armed = self.t
        return True, 'armed'

    def disarm(self):
        self.armed = False

    # ------------------------------------------------------------------
    #  decisions: resolve, check, fit, filter, apply
    # ------------------------------------------------------------------
    def offset_limits(self, slots):
        """Range of offsets (in the route's + direction, metres) for which every
        lane of this formation stays on the floor and clear of the walls over
        the part of the route the offset moves."""
        key = tuple(sorted(set(round(l, 3) for _, l in slots)))
        if key in self._limits_cache:
            return self._limits_cache[key]
        r = self.route
        span = getattr(r, 'span', None)
        if span is None:
            self._limits_cache[key] = (0.0, 0.0)
            return 0.0, 0.0
        keep = r.offset
        clear = self.p.body_radius + self.p.fit_margin
        lanes = F.lanes_of(slots)

        work = [lg for lg in r.legs if lg['work']]

        def ok(off):
            r.set_offset(off)
            # the leader's own line over everything the offset moves (turns are
            # driven in single file), and every lane along the working legs
            checks = [(0.0, span[0], span[1])]
            for lg in work:
                checks += [(lanes[0], lg['s0'], lg['s1']), (lanes[-1], lg['s0'], lg['s1'])]
            for lat, a, b in checks:
                pts, _ = r.lane_samples(lat, a, b, 0.06)
                good, _ = self.arena.points_free(pts, clear, margin=self.p.body_radius + 0.02)
                if not good:
                    return False
            return True
        step = 0.05
        lo = hi = 0.0
        if not ok(0.0):
            r.set_offset(keep)
            self._limits_cache[key] = (0.0, -1.0)            # empty range: this formation has no valid offset
            return 0.0, -1.0
        k = 1
        while k * step <= self.p.max_offset + 1e-9 and ok(k * step):
            hi = k * step
            k += 1
        k = 1
        while k * step <= self.p.max_offset + 1e-9 and ok(-k * step):
            lo = -k * step
            k += 1
        r.set_offset(keep)
        self._limits_cache[key] = (lo, hi)
        return lo, hi

    def _track_coords(self, states):
        """(along, lat) of each follower relative to the leader, in the track frame."""
        out = []
        for name in self.followers:
            st = states.get(name)
            if st is None:
                out.append((-self.p.spacing * (len(out) + 1), 0.0))
                continue
            s, e = self.route.project(st.x, st.y, self.s.get(name))
            out.append((s - self.s_leader, e))
        return out

    def fits(self, slots, states, perm=None):
        """Can this formation be taken up here and now? Returns (ok, reason).

        It fits when (1) every lane stays on the floor and clear of the walls
        from the tail of the crew to fit_reach ahead of the leader, (2) no lane
        is asked to turn tighter than a rover can, or to go faster than a rover
        can, within that stretch, and (3) every follower can drive straight to
        its new slot without crossing a wall. Free-standing obstacles are not
        part of the check: they are the reflex's job (the paper's fit check 2)."""
        p, r = self.p, self.route
        if not slots:
            return True, ''
        cur = self._track_coords(states)
        tail = min([a for a, _ in slots] + [a for a, _ in cur] + [0.0])
        s0 = max(r.s[0], self.s_leader + tail - 0.1)
        s1 = min(r.s[-1], self.s_leader + p.fit_reach)
        clear = p.body_radius + p.fit_margin
        lats = sorted(set(round(l, 3) for _, l in slots if abs(l) > 1e-6))
        for lat in lats:
            pts, _ = r.lane_samples(lat, s0, s1, 0.05)
            ok, why = self.arena.points_free(pts, clear, margin=p.body_radius + 0.02)
            if not ok:
                return False, 'lane %+.2f m %s' % (lat, why)
        kmax, kmin = r.max_curvature(s0, s1)
        for lat in lats:
            for k in (kmax, kmin):
                if abs(k) < 1e-3:
                    continue
                den = 1.0 - lat * k
                if den <= 0.05 or abs(k) / den > 1.0 / p.min_turn_radius + 1e-6:
                    return False, 'lane %+.2f m would have to turn tighter than %.2f m' % (lat, p.min_turn_radius)
                if self.speed * den > p.v_max * 1.02:
                    return False, 'lane %+.2f m would have to go faster than %.2f m/s in the turn' % (lat, p.v_max)
        if perm is None:
            perm = F.assign(cur, slots)
        for i, name in enumerate(self.followers):
            st = states.get(name)
            if st is None:
                continue
            a, l = slots[perm[i]]
            x, y, _, _ = r.lane_point(self.s_leader + a, l)
            if self.arena.segment_wall_distance((st.x, st.y), (x, y)) < p.body_radius + 0.03:
                return False, '%s would cross a wall to reach its place' % name
        return True, ''

    def formations_that_fit(self, states):
        return {name: bool(self.fits(F.preset_slots(name, self.n, self.p.spacing), states)[0]) for name in F.PRESETS}

    def _shield(self, name, slots, states):
        """The safety filter: the fit check made binding (the paper's filter,
        final version). Returns (name, slots, held)."""
        if self.fits(slots, states)[0]:
            return name, slots, False
        h = F.half_width(slots)
        if h == 0 or h < F.half_width(self.slots_target):
            return name, slots, False                        # narrowing is never held back
        if self.fits(self.slots_target, states)[0]:
            return self.formation, list(self.slots_target), True
        best = None
        for k in F.PRESETS:
            sl = F.preset_slots(k, self.n, self.p.spacing)
            if self.fits(sl, states)[0]:
                w = F.lanes_of(sl)
                if best is None or (w[-1] - w[0]) > best[0]:
                    best = (w[-1] - w[0], k, sl)
        if best is not None:
            return best[1], best[2], True
        return 'column', F.preset_slots('column', self.n, self.p.spacing), True

    def request(self, decision, states, source='decider'):
        """A decider's choice: dict(formation, speed, offset). Read leniently,
        checked strictly, then passed through the safety filter. Returns a
        record of what happened; record['error'] is set if the reply was
        unusable (and the previous choice stays in force)."""
        p = self.p
        rec = dict(t=round(self.t, 2), source=source, asked=_jsonable(decision))
        try:
            if not isinstance(decision, dict):
                raise ValueError('reply is not a JSON object')
            f = decision.get('formation')
            if isinstance(f, dict) and 'formation' in f and 'slots' not in f and 'shape' not in f:
                # {"formation": {"formation": "column", "speed": ...}}: unwrap
                decision = dict(decision, **{k: v for k, v in f.items() if k in ('speed', 'offset') and k not in decision})
                f = f['formation']
            if f is None:
                raise ValueError('no "formation" in the reply')
            name, slots = F.resolve(f, self.n, p.spacing, min_gap=max(p.protected, 2 * p.body_radius + 0.05),
                                    max_lat=3 * p.spacing * max(1, self.n), max_along=3 * p.spacing * max(1, self.n))
            sp = decision.get('speed', self.speed)
            sp = self.speed if sp is None else float(sp)
            if not math.isfinite(sp):
                raise ValueError('speed is not a number')
            off = decision.get('offset', self.offset_target)   # not mentioned: stays as it is
            off = self.offset_target if off is None else float(off)
            if not math.isfinite(off):
                raise ValueError('offset is not a number')
        except Exception as e:
            rec['error'] = str(e)
            self._rejected = str(e)
            self.log.append(rec)
            return rec
        self._rejected = None
        sp = max(p.speed_range[0], min(p.speed_range[1], sp))
        name2, slots2, held = self._shield(name, slots, states)
        if held:
            self.wanted = dict(name=name, slots=slots, speed=sp, offset=off)
            rec['held'] = name
        else:
            self.wanted = None
        self._apply(name2, slots2, sp, off, states)
        rec['applied'] = dict(formation=self.formation, speed=round(self.speed, 3), offset=round(self.offset_target, 3))
        self.log.append(rec)
        return rec

    def _apply(self, name, slots, speed, offset, states):
        changed = [round(v, 3) for s in slots for v in s] != [round(v, 3) for s in self.slots_target for v in s]
        self.formation = name
        self.speed = speed
        lo, hi = self.offset_limits(slots)
        self.offset_target = max(lo, min(hi, offset)) if lo <= hi else 0.0
        if changed or len(self.assign) != self.n:
            cur = self._track_coords(states)
            self.slots_target = list(slots)
            self.assign = F.assign(cur, slots) if self.n else []
            if not self.armed:
                self.slots_now = [slots[self.assign[i]] for i in range(self.n)]

    def _enforce_fit(self, states):
        """A formation that has stopped fitting is not left in force while a
        decider thinks about it (or has crashed): after one second of not
        fitting, the crew takes the widest formation that does fit, single
        file if none, and the displaced choice is held to be taken up again
        when it fits. The fit check looks fit_reach ahead, so this happens
        well before the gate or the turn."""
        if self.t - self._fit_checked < 0.5:
            return
        self._fit_checked = self.t
        if self.fits(self.slots_target, states)[0]:
            self._misfit_since = None
            return
        if self._misfit_since is None:
            self._misfit_since = self.t
            return
        if self.t - self._misfit_since < 1.0:
            return
        prev = dict(name=self.formation, slots=list(self.slots_target), speed=self.speed, offset=self.offset_target)
        best = None
        for k in F.PRESETS:
            sl = F.preset_slots(k, self.n, self.p.spacing)
            if F.half_width(sl) < F.half_width(self.slots_target) and self.fits(sl, states)[0]:
                w = F.lanes_of(sl)
                if best is None or (w[-1] - w[0]) > best[0]:
                    best = (w[-1] - w[0], k, sl)
        name, slots = (best[1], best[2]) if best else ('column', F.preset_slots('column', self.n, self.p.spacing))
        if F.half_width(slots) >= F.half_width(self.slots_target):
            return                                            # already as narrow as it gets
        if self.wanted is None:
            self.wanted = prev
        self._apply(name, slots, self.speed, self.offset_target, states)
        self._misfit_since = None
        self.log.append(dict(t=round(self.t, 2), source='safety filter', narrowed_to=name, held=prev['name']))

    def _release_wanted(self, states):
        w = self.wanted
        if w is None:
            return
        if self.fits(w['slots'], states)[0]:
            self.wanted = None
            self._apply(w['name'], w['slots'], w['speed'], w['offset'], states)
            self.log.append(dict(t=round(self.t, 2), released=w['name']))

    # ------------------------------------------------------------------
    #  decision points and the observation a decider reads
    # ------------------------------------------------------------------
    def phase(self):
        if self.done:
            return 'done'
        return self.route.leg_at(self.s_leader)['name']

    def on_work_leg(self):
        return bool(self.route.leg_at(self.s_leader)['work']) and not self.done

    def hazards_in_range(self, states):
        st = states.get(self.leader)
        out = []
        if st is None:
            return out
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        for i, o in enumerate(self.arena.obstacles):
            dx, dy = o['x'] - st.x, o['y'] - st.y
            if math.hypot(dx, dy) <= self.p.sensor_range + o['r']:
                out.append(dict(id=i, type='rock', name=o['name'], aheadM=round(dx * c + dy * s, 2),
                                leftM=round(-dx * s + dy * c, 2), radiusM=round(o['r'], 2),
                                x=round(o['x'], 2), y=round(o['y'], 2), note='solid'))
        return out

    def decision_due(self, states):
        """Reasons to consult the decider now ([] if none): the start, a new
        phase, a hazard coming into range, the formation no longer fitting, a
        wider formation having fitted for a while, and otherwise every
        `period` seconds."""
        tr, p = self._trig, self.p
        why = []
        if not tr['started']:
            tr['started'] = True
            tr['phase'] = self.phase()
            tr['last'] = self.t
            tr['seen'] = {h['id'] for h in self.hazards_in_range(states)}
            return ['start']
        if not self.armed or self.done:
            return why
        if self.t - tr['checked'] < 0.5:
            return why
        tr['checked'] = self.t
        ph = self.phase()
        if ph != tr['phase']:
            why.append('entering %s' % ph)
            tr['offered'] = set()
        new = [h['id'] for h in self.hazards_in_range(states) if h['id'] not in tr['seen']]
        if new:
            why.append('hazard in sensor range')
        tr['seen'] |= set(new)
        fit_now = self.fits(self.slots_target, states)[0]
        tr['misfit'] = 0 if fit_now else tr['misfit'] + 1
        if tr['misfit'] == 2:
            why.append('formation no longer fits')
        n_cur = len(F.lanes_of(self.slots_target))
        for name in F.PRESETS:
            sl = F.preset_slots(name, self.n, p.spacing)
            if self.fits(sl, states)[0]:
                tr['since'].setdefault(name, self.t)
            else:
                tr['since'].pop(name, None)
        room = [k for k, t0 in tr['since'].items() if self.t - t0 >= p.room_for and k not in tr['offered']
                and len(F.lanes_of(F.preset_slots(k, self.n, p.spacing))) > n_cur]
        if room and self.t - tr['last'] >= 3.0:
            why.append('room for a wider formation')
            tr['offered'] |= set(room)
        if self.t - tr['last'] >= p.period:
            why.append('periodic')
        tr['phase'] = ph
        if why:
            tr['last'] = self.t
        return why

    def observation(self, states, why, instruction='', metrics=None):
        """What the decider is told. Same shape as shepherd's farm observation,
        in metres on this floor."""
        p, r, ar = self.p, self.route, self.arena
        st = states.get(self.leader)
        leg = r.leg_at(self.s_leader)
        lanes = F.lanes_of(self.slots_target)
        plus, minus = ar.offset_words(r)
        errs = [e for e in self.status.get('slot_error', {}).values()]
        hz = self.hazards_in_range(states)
        ob = dict(
            instruction=instruction,
            calledBecause=list(why),
            timeS=round(self.t - self.t_armed if self.armed else 0.0, 0),
            phase='%s%s' % (leg['name'], ' (working pass)' if leg['work'] else ''),
            leader=dict(x=round(st.x, 2), y=round(st.y, 2), heading=ar.compass(st.yaw),
                        speed=round(self.speed, 2)) if st is not None else None,
            formation=dict(current=self.formation, lanes=lanes, lengthM=round(F.length_of(self.slots_target), 2),
                           meanDistanceOfFollowersFromSlotsM=round(float(np.mean(errs)), 2) if errs else None),
            hazardsInRange=hz,
        )
        if st is not None:
            x, y, th, _ = r.at(self.s_leader)
            left = th + math.pi / 2
            # + offset is along route.axis; lanes are left of travel. sg converts between them.
            sg = 1.0 if (math.cos(left) * r.axis[0] + math.sin(left) * r.axis[1]) >= 0 else -1.0
            for h in hz:
                o = ar.obstacles[h['id']]
                _, e = r.project(o['x'], o['y'], self.s_leader, back=0.5, ahead=p.sensor_range + 0.5)
                touch = [l for l in lanes if abs(e - l) < o['r'] + p.body_radius + p.reflex_margin]
                h['yourLanesThatWouldTouchIt'] = [round(l, 2) for l in touch]
            in_field = bool(leg['work']) or float(r.w[r.index(min(max(self.s_leader, 0.0), r.length))]) > 0.5
            if in_field:
                allowed = {}
                for k in F.PRESETS:
                    lo, hi = self.offset_limits(F.preset_slots(k, self.n, p.spacing))
                    if lo <= hi:
                        allowed[k] = [round(lo, 2) + 0.0, round(hi, 2) + 0.0]
                ob['pass'] = dict(working=bool(leg['work']), heading=ar.compass(th), leftOfTravelIs=ar.compass(left),
                                  metresDone=round(self.s_leader - leg['s0'], 2),
                                  metresLeft=round(leg['s1'] - self.s_leader, 2),
                                  offsetMeaning='+ moves the working line %s, - moves it %s' % (plus, minus),
                                  currentOffsetM=round(self.offset_target, 2) + 0.0,
                                  lanesAreLeftOfTravelWhen='lane > 0',
                                  aPositiveLaneIsToThe=plus if sg > 0 else minus,
                                  allowedOffsetM=allowed)
        ob['formationsThatFit'] = self.formations_that_fit(states)
        if metrics is not None:
            ob['progress'] = metrics.brief()
        if self.wanted is not None:
            ob['heldBack'] = '%s does not fit here yet; it will be taken up when it does' % self.wanted['name']
        if self._rejected:
            ob['lastReplyRejected'] = self._rejected
        return ob

    # ------------------------------------------------------------------
    #  the obstacle reflex (committed side)
    # ------------------------------------------------------------------
    def _reflex(self, name, st, chi, v, states):
        """Bend the wanted course `chi` round any free-standing obstacle that
        lies on it. Returns (course, speed, active)."""
        p = self.p
        active = False
        chi0 = chi
        psi = st.yaw
        order = sorted(range(len(self.arena.obstacles)),
                       key=lambda i: math.hypot(self.arena.obstacles[i]['x'] - st.x, self.arena.obstacles[i]['y'] - st.y))
        for i in order:
            o = self.arena.obstacles[i]
            dx, dy = o['x'] - st.x, o['y'] - st.y
            dist = max(math.hypot(dx, dy), 1e-6)
            key = (name, i)
            if dist - o['r'] > p.reflex_look:
                self.side.pop(key, None)                      # out of reach: the commitment ends
                continue
            b = math.atan2(dy, dx)
            half = math.asin(min(1.0, (o['r'] + p.body_radius + p.reflex_margin) / dist))
            beta = wrap(b - psi)                              # obstacle relative to the actual heading
            if abs(beta) > math.radians(100.0):
                self.side.pop(key, None)                      # behind: done with it
                continue
            dW = wrap(chi - b)                                # wanted course relative to the obstacle
            delta = wrap(chi - psi)                           # the turn being asked for
            sweep = (beta * delta > 0) and (abs(beta) < abs(delta))
            s = self.side.get(key, 0)
            if s == 0 and (abs(dW) < half or abs(beta) < half or sweep):
                cl, cr = self._crowding(name, st, states)
                cost = {}
                for sd, crowd in ((1, cl), (-1, cr)):         # +1: pass with the obstacle on the right
                    edge = b + sd * half
                    cost[sd] = abs(wrap(edge - psi)) + 0.5 * abs(wrap(edge - chi)) + 0.5 * crowd
                s = 1 if cost[1] <= cost[-1] else -1
            if s != 0 and s * wrap(psi - b) < -min(half, math.radians(10.0)):
                s = -s                                        # it has ended up on the other side: accept that
            if s != 0:
                self.side[key] = s
                if abs(dW) < half or (s * dW < 0 and sweep):
                    fade = max(0.0, min(1.0, (p.reflex_look - (dist - o['r'])) / (p.reflex_look / 2)))
                    chi = chi + fade * wrap(b + s * half - chi)
                    active = True
        if active:
            wgt = min(1.0, abs(wrap(chi - chi0)) / (math.pi / 2))
            v = v * (1.0 - 0.5 * wgt)
        return chi, v, active

    def _crowding(self, name, st, states):
        left = right = 0.0
        for other, so in states.items():
            if other == name:
                continue
            dx, dy = so.x - st.x, so.y - st.y
            d = math.hypot(dx, dy)
            if d > 1.5 or d < 1e-6:
                continue
            lat = -dx * math.sin(st.yaw) + dy * math.cos(st.yaw)
            w = (1.0 - d / 1.5) * abs(lat) / d
            if lat > 0:
                left += w
            else:
                right += w
        return left, right

    def _committed_side(self, name):
        """Side of the nearest obstacle this rover is committed to passing on, or 0."""
        best = (float('inf'), 0)
        for (n, i), s in self.side.items():
            if n == name and s != 0:
                best = min(best, (i, s))
        return best[1]

    # ------------------------------------------------------------------
    #  one control cycle
    # ------------------------------------------------------------------
    def step(self, states, dt, now=None):
        """states: dict name -> RobotState (age = seconds since the pose was
        taken). Returns dict name -> (v, w). Fills self.status."""
        p, r = self.p, self.route
        dt = max(0.0, min(float(dt), 0.5))
        self.t += dt
        zero = {n: (0.0, 0.0) for n in self.names}
        self.stop_cause = {}
        self.fleet_stop = None

        # ---- stops that apply to the whole fleet ----
        for n in self.names:
            st = states.get(n)
            if st is None:
                self.fleet_stop = 'no pose for %s' % n
                break
            if not (st.age >= -0.05 and st.age <= p.stale_timeout):
                self.fleet_stop = 'stale pose for %s (%.2f s)' % (n, st.age)
                break
            if not all(math.isfinite(v) for v in (st.x, st.y, st.yaw)):
                self.fleet_stop = 'bad pose for %s' % n
                break
        if self.fleet_stop is None:
            for n in self.names:
                st = states[n]
                if not self.arena.inside(st.x, st.y, -p.geofence_slack):
                    self.latched = 'geofence: %s left the floor area' % n
                    self.armed = False
        if self.latched:
            self.fleet_stop = self.latched
        if self.fleet_stop is not None or not self.armed or self.done:
            self._fill_status(states, zero)
            return zero

        # ---- progress and the moving reference ----
        lead = states[self.leader]
        if self.leader not in self.s:
            s0, _ = r.project(lead.x, lead.y, 0.0, back=min(1.0, -r.s[0]), ahead=2.0)
            self.s[self.leader] = s0
            self.s_leader = s0
        d_off = self.offset_target - self.offset_applied
        if abs(d_off) > 1e-6:
            self.offset_applied += max(-p.offset_rate * dt, min(p.offset_rate * dt, d_off))
            r.set_offset(self.offset_applied)
        sL, eL = r.project(lead.x, lead.y, self.s_leader)
        self.s_leader = max(self.s_leader, sL)
        self.s[self.leader] = self.s_leader
        self._release_wanted(states)
        self._enforce_fit(states)

        # slots drift to their assigned places
        for i in range(self.n):
            ta, tl = self.slots_target[self.assign[i]]
            a, l = self.slots_now[i]
            step = p.slot_rate * dt
            self.slots_now[i] = (a + max(-step, min(step, ta - a)), l + max(-step, min(step, tl - l)))

        cmds = {}
        slot_err = {}
        # ---- leader ----
        leader_done = self.s_leader >= r.length - p.goal_tolerance
        lw0, lw1, lwf = p.leader_wait
        wait = max(lwf, min(1.0, 1.0 - (self._lag - lw0) / max(lw1 - lw0, 1e-6)))
        x, y, th, kap = r.at(self.s_leader)
        if leader_done:
            cmds[self.leader] = (0.0, 0.0)
            self._leader_cmd_v = 0.0
        else:
            chi = th - p.chi_inf_leader * (2 / math.pi) * math.atan(self.k_cross * eL)
            v = self.speed * wait
            # slow for the last few centimetres so the stop is on the mark
            v = min(v, max(0.04, 1.5 * (r.length - self.s_leader)))
            chi, v, act = self._reflex(self.leader, lead, chi, v, states)
            w = p.k_heading * wrap(chi - lead.yaw) + (0.0 if act else v * kap)
            cmds[self.leader] = (v, w)
            self._leader_cmd_v = v

        # ---- followers ----
        lag = 0.0
        for i, name in enumerate(self.followers):
            st = states[name]
            a, l = self.slots_now[i]
            s_slot = self.s_leader + a
            if name not in self.s:
                self.s[name], _ = r.project(st.x, st.y, s_slot, back=1.5, ahead=1.5)
            sf, df = r.project(st.x, st.y, self.s[name], back=0.4, ahead=0.8)
            self.s[name] = sf
            _, _, thf, kf = r.at(sf)
            den = max(0.3, 1.0 - l * kf)
            e_lat = df - l
            e_al = (sf - s_slot) * den
            sx, sy, _, _ = r.lane_point(s_slot, l)
            slot_err[name] = math.hypot(st.x - sx, st.y - sy)
            lag = max(lag, -e_al)
            chi = thf - p.chi_inf_follower * (2 / math.pi) * math.atan(self.k_cross * e_lat)
            v_ref = self._leader_cmd_v * den
            v_des = max(0.0, min(p.v_max, v_ref - p.k_along * e_al))
            if leader_done and e_al > -p.goal_tolerance:
                v_des = 0.0                                   # in its place at the end: stay
            vx, vy = v_des * math.cos(chi), v_des * math.sin(chi)
            # separation, inverse-distance profile
            px = py = 0.0
            for other in self.names:
                if other == name:
                    continue
                so = states[other]
                dx, dy = st.x - so.x, st.y - so.y
                d = math.hypot(dx, dy)
                if 1e-6 < d < p.protected:
                    g = p.k_sep * (1.0 / d - 1.0 / p.protected)
                    px += g * dx / d
                    py += g * dy / d
            # the reflex owns the turn: a flock push that would turn this rover
            # toward the obstacle it is passing is dropped
            sd = self._committed_side(name)
            if sd != 0 and (px or py):
                nx, ny = -math.sin(st.yaw), math.cos(st.yaw)
                latp = px * nx + py * ny                      # + turns it left
                if (latp > 0) != (sd > 0):
                    px -= latp * nx
                    py -= latp * ny
            vx += px
            vy += py
            sp = math.hypot(vx, vy)
            if sp < 1e-4:
                cmds[name] = (0.0, 0.0)
                continue
            chi_c = math.atan2(vy, vx)
            v = min(p.v_max, sp) * (0.4 + 0.6 * max(0.0, math.cos(wrap(chi_c - st.yaw))))
            if v_des <= 1e-6 and (px or py):
                v = min(v, 0.06)                              # nudged while parked: creep, don't lunge
            chi_c, v, act = self._reflex(name, st, chi_c, v, states)
            w = p.k_heading * wrap(chi_c - st.yaw) + (0.0 if act else v * kf / den)
            # ranked braking: slow for any higher-ranked machine ahead and close
            g = 1.0
            for other in self.names[:i + 1]:
                so = states[other]
                dx, dy = so.x - st.x, so.y - st.y
                d = math.hypot(dx, dy)
                if d < p.brake_start and abs(wrap(math.atan2(dy, dx) - st.yaw)) < math.radians(70.0):
                    g = min(g, max(0.0, (d - p.brake_stop) / max(p.brake_start - p.brake_stop, 1e-6)))
            if g < 1.0:
                v, w = v * g, w * g                           # together: same curvature
                if g <= 0.0:
                    self.stop_cause[name] = 'braking for a machine ahead'
            cmds[name] = (v, w)
        self._lag = lag

        # ---- stops that apply to one machine ----
        for name in self.names:
            st = states[name]
            v, w = cmds[name]
            if v <= 0.0:
                continue
            hx, hy = math.cos(st.yaw), math.sin(st.yaw)
            for other in self.names:
                if other == name:
                    continue
                so = states[other]
                dx, dy = so.x - st.x, so.y - st.y
                d = math.hypot(dx, dy)
                if d - 2 * p.body_radius < p.stop_gap and (dx * hx + dy * hy) / max(d, 1e-6) > math.cos(math.radians(50.0)):
                    cmds[name] = (0.0, 0.0)
                    self.stop_cause[name] = 'stopped: %s is in the way' % other
            dw, wall = self.arena.nearest_wall(st.x, st.y)
            if wall is not None and dw < p.body_radius + 0.04:
                ax, ay = wall['a']
                bx, by = wall['b']
                ux, uy = bx - ax, by - ay
                L2 = max(ux * ux + uy * uy, 1e-12)
                t = max(0.0, min(1.0, ((st.x - ax) * ux + (st.y - ay) * uy) / L2))
                cx, cy = ax + t * ux - st.x, ay + t * uy - st.y
                if (cx * hx + cy * hy) / max(math.hypot(cx, cy), 1e-6) > 0.3:
                    cmds[name] = (0.0, 0.0)
                    self.stop_cause[name] = 'stopped: at %s' % wall['name']
            for o in self.arena.obstacles:
                dx, dy = o['x'] - st.x, o['y'] - st.y
                d = math.hypot(dx, dy)
                if d - o['r'] - p.body_radius < 0.03 and (dx * hx + dy * hy) / max(d, 1e-6) > 0.5:
                    cmds[name] = (0.0, 0.0)
                    self.stop_cause[name] = 'stopped: at %s' % o['name']

        out = {n: limit_command(cmds[n][0], cmds[n][1], p) for n in self.names}
        # ---- finished? ----
        if leader_done and all(out[n][0] == 0.0 for n in self.names):
            settled = all((self.s[n] - (self.s_leader + self.slots_now[i][0])) > -3 * p.goal_tolerance or
                          n in self.stop_cause for i, n in enumerate(self.followers))
            if settled:
                self.done = True
                self.armed = False
        self._fill_status(states, out, slot_err)
        return out

    def _fill_status(self, states, cmds, slot_err=None):
        self.status = dict(
            t=round(self.t, 2), armed=self.armed, done=self.done, phase=self.phase() if self.leader in self.s else 'not started',
            fleet_stop=self.fleet_stop, stops=dict(self.stop_cause), formation=self.formation,
            speed=round(self.speed, 3), offset=round(self.offset_applied, 3), offset_target=round(self.offset_target, 3),
            held=self.wanted['name'] if self.wanted else None, s_leader=round(self.s_leader, 3),
            route_length=round(self.route.length, 3), slot_error={k: round(v, 3) for k, v in (slot_err or {}).items()},
            cmd={n: (round(c[0], 3), round(c[1], 3)) for n, c in cmds.items()},
            age={n: round(states[n].age, 3) for n in self.names if n in states})

    # ------------------------------------------------------------------
    #  where should the rovers be put down?
    # ------------------------------------------------------------------
    def start_poses(self):
        """Where to place each rover before arming: the leader on the start
        of the route, followers in single file behind it. dict name -> (x, y, yaw)."""
        out = {}
        x, y, th, _ = self.route.at(0.0)
        out[self.leader] = (x, y, th)
        for i, name in enumerate(self.followers):
            x, y, th, _ = self.route.at(-(i + 1) * self.p.spacing)
            out[name] = (x, y, th)
        return out

    def slot_positions(self):
        """Current slot of each follower in arena coordinates (for display)."""
        out = {}
        for i, name in enumerate(self.followers):
            a, l = self.slots_now[i]
            x, y, th, _ = self.route.lane_point(self.s_leader + a, l)
            out[name] = (x, y, th)
        return out


def _jsonable(o):
    try:
        import json
        json.dumps(o)
        return o
    except Exception:
        return repr(o)
