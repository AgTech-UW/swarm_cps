"""A small world to rehearse in: kinematic Ackermann rovers with actuator lag,
a camera that reports noisy, late, occasionally missing poses, and the scoring
used to compare deciders. Nothing here is used when real rovers are driving,
except Metrics, which scores real runs the same way.
"""
import math

import numpy as np

from .controller import RobotState, SwarmController


class Metrics:
    """Collisions, clearances, time in each named zone, and ground worked.
    update() takes positions, so it scores a simulated or a real run alike.
    From camera poses a "collision" is a disc overlap as seen by the camera,
    not a certified touch."""

    def __init__(self, arena, route, body_radius, cell=0.05):
        self.arena, self.route, self.r = arena, route, body_radius
        self.t = 0.0
        self.zone_s = {z['name']: 0.0 for z in arena.zones}
        self.collisions = []
        self._touch = set()
        self.min_clearance = float('inf')
        self.stopped_s = 0.0
        # the field: the floor either side of the working legs
        work = [lg for lg in route.legs if lg['work']]
        self.cell = cell
        if work:
            pts = np.concatenate([route.lane_samples(0.0, lg['s0'], lg['s1'], 0.05)[0] for lg in work])
            pad = 0.9
            x0, x1 = max(arena.x0, pts[:, 0].min() - 0.15), min(arena.x1, pts[:, 0].max() + 0.15)
            y0, y1 = max(arena.y0, pts[:, 1].min() - pad), min(arena.y1, pts[:, 1].max() + pad)
            if abs(route.axis[0]) > 0.5:
                x0, x1 = max(arena.x0, pts[:, 0].min() - pad), min(arena.x1, pts[:, 0].max() + pad)
                y0, y1 = max(arena.y0, pts[:, 1].min() - 0.15), min(arena.y1, pts[:, 1].max() + 0.15)
            f = arena.cfg.get('field')
            if f:
                x0, x1, y0, y1 = float(f['x'][0]), float(f['x'][1]), float(f['y'][0]), float(f['y'][1])
            self.field = (x0, x1, y0, y1)
            gx, gy = np.meshgrid(np.arange(x0 + cell / 2, x1, cell), np.arange(y0 + cell / 2, y1, cell))
            self.gx, self.gy = gx, gy
            self.worked = np.zeros(gx.shape, bool)
            blocked = np.zeros(gx.shape, bool)
            for o in arena.obstacles:
                blocked |= np.hypot(gx - o['x'], gy - o['y']) < o['r']
            self.workable = ~blocked
        else:
            self.field = None
            self.worked = self.workable = np.zeros((1, 1), bool)

    def update(self, pos, dt, working=(), stopped=()):
        """pos: dict name -> (x, y). working: names currently on a working leg."""
        self.t += dt
        names = list(pos)
        for n in names:
            x, y = pos[n]
            for z in self.arena.zones_at(x, y):
                self.zone_s[z] += dt
        self.stopped_s += dt * len(stopped)
        now = set()
        for i in range(len(names)):
            xi, yi = pos[names[i]]
            for j in range(i + 1, len(names)):
                xj, yj = pos[names[j]]
                gap = math.hypot(xi - xj, yi - yj) - 2 * self.r
                self.min_clearance = min(self.min_clearance, gap)
                if gap < 0:
                    now.add(('rover', names[i], names[j]))
            for o in self.arena.obstacles:
                gap = math.hypot(xi - o['x'], yi - o['y']) - o['r'] - self.r
                self.min_clearance = min(self.min_clearance, gap)
                if gap < 0:
                    now.add(('obstacle', names[i], o['name']))
            dw, wall = self.arena.nearest_wall(xi, yi)
            if wall is not None:
                self.min_clearance = min(self.min_clearance, dw - self.r)
                if dw - self.r < 0:
                    now.add(('wall', names[i], wall['name']))
        for c in now - self._touch:                           # a contact counts once, when it starts
            self.collisions.append(dict(t=round(self.t, 1), kind=c[0], a=c[1], b=c[2]))
        self._touch = now
        if self.field is not None:
            half = self.arena.swath / 2
            for n in working:
                x, y = pos[n]
                i0 = int((y - half - self.field[2]) / self.cell) - 1
                i1 = int((y + half - self.field[2]) / self.cell) + 2
                j0 = int((x - half - self.field[0]) / self.cell) - 1
                j1 = int((x + half - self.field[0]) / self.cell) + 2
                i0, j0 = max(i0, 0), max(j0, 0)
                if i1 <= i0 or j1 <= j0:
                    continue
                sub = (self.gx[i0:i1, j0:j1] - x) ** 2 + (self.gy[i0:i1, j0:j1] - y) ** 2 < half * half
                self.worked[i0:i1, j0:j1] |= sub

    def coverage(self):
        n = int(self.workable.sum())
        return float((self.worked & self.workable).sum()) / n if n else 0.0

    def brief(self):
        return dict(fieldWorkedPct=round(100 * self.coverage(), 1),
                    machineSecondsInZones={k: round(v, 0) for k, v in self.zone_s.items()},
                    collisionsSoFar=len(self.collisions))

    def result(self):
        return dict(time_s=round(self.t, 1), collisions=len(self.collisions), collision_log=list(self.collisions),
                    min_clearance_m=round(self.min_clearance, 3) if math.isfinite(self.min_clearance) else None,
                    zone_machine_seconds={k: round(v, 1) for k, v in self.zone_s.items()},
                    field_worked_pct=round(100 * self.coverage(), 1), stopped_machine_seconds=round(self.stopped_s, 1))


class SimWorld:
    """Rovers that respond like hardware: first-order lag on speed and yaw
    rate, a hard turning-radius limit that may differ from what the controller
    assumes, a motor-side command timeout, and a camera with noise, delay and
    dropped frames."""

    def __init__(self, names, poses, true_turn_radius=0.35, tau_v=0.25, tau_w=0.20, cmd_timeout=0.5,
                 pos_noise=0.004, yaw_noise=math.radians(1.0), latency=0.08, dropout=0.02, v_scale=None, seed=0):
        self.names = list(names)
        self.state = {n: [poses[n][0], poses[n][1], poses[n][2], 0.0, 0.0] for n in names}   # x, y, yaw, v, w
        self.cmd = {n: (0.0, 0.0, -1e9) for n in names}
        self.R = true_turn_radius
        self.tau_v, self.tau_w = tau_v, tau_w
        self.cmd_timeout = cmd_timeout
        self.pos_noise, self.yaw_noise = pos_noise, yaw_noise
        self.latency, self.dropout = latency, dropout
        self.v_scale = v_scale or {}
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.hist = []                                        # (t, snapshot) for the delayed camera
        self.last_meas = {}
        self.frozen = set()                                   # rovers with a dead motor (fault injection)
        self.blind = set()                                    # rovers the camera cannot see

    def command(self, cmds):
        for n, (v, w) in cmds.items():
            self.cmd[n] = (float(v), float(w), self.t)

    def step(self, dt):
        sub = max(1, int(math.ceil(dt / 0.01)))
        h = dt / sub
        for _ in range(sub):
            for n in self.names:
                x, y, yaw, v, w = self.state[n]
                cv, cw, tc = self.cmd[n]
                if self.t - tc > self.cmd_timeout or n in self.frozen:
                    cv = cw = 0.0                             # the motor driver's own watchdog
                cv = max(0.0, cv) * self.v_scale.get(n, 1.0)
                v += (cv - v) * min(1.0, h / self.tau_v)
                lim = abs(v) / self.R                         # Ackermann: no speed, no turn
                cw = max(-lim, min(lim, cw))
                w += (cw - w) * min(1.0, h / self.tau_w)
                w = max(-lim, min(lim, w))
                x += v * math.cos(yaw) * h
                y += v * math.sin(yaw) * h
                yaw += w * h
                self.state[n] = [x, y, yaw, v, w]
            self.t += h
        self.hist.append((self.t, {n: tuple(self.state[n]) for n in self.names}))
        while len(self.hist) > 2 and self.hist[1][0] < self.t - self.latency - 0.2:
            self.hist.pop(0)

    def truth(self):
        return {n: (self.state[n][0], self.state[n][1]) for n in self.names}

    def measure(self):
        """What the camera pipeline would publish now: dict name -> RobotState
        with `age` set, from the scene as it was `latency` ago."""
        tq = self.t - self.latency
        snap = None
        for t, s in self.hist:
            if t <= tq + 1e-9:
                snap = (t, s)
        if snap is not None:
            for n in self.names:
                if n in self.blind or self.rng.random() < self.dropout:
                    continue
                x, y, yaw, v, w = snap[1][n]
                self.last_meas[n] = (snap[0], x + self.rng.normal(0, self.pos_noise), y + self.rng.normal(0, self.pos_noise),
                                     yaw + self.rng.normal(0, self.yaw_noise), v)
        out = {}
        for n, (t0, x, y, yaw, v) in self.last_meas.items():
            out[n] = RobotState(x, y, yaw, v, age=self.t - t0)
        return out


def run_mission(arena, names, decider=None, instruction='', params=None, sim=None, dt=1 / 30.0, t_max=240.0,
                on_step=None, decision_latency=0.0, start_poses=None, events=None):
    """Run one mission in simulation. decider: object with decide(observation)
    -> dict or None (None keeps the current choice); None means single file
    throughout. decision_latency delays each reply by that many seconds while
    the rovers keep driving, as a slow model would. events: [(time, fn(world,
    controller))] for fault injection. Returns dict(result, log, transcript,
    controller, world)."""
    ctl = SwarmController(arena, names, params)
    poses = start_poses or ctl.start_poses()
    world = sim or SimWorld(names, poses, true_turn_radius=ctl.p.min_turn_radius)
    met = Metrics(arena, ctl.route, ctl.p.body_radius)
    transcript = []
    pending = []                                              # (due time, decision, observation)
    events = sorted(events or [], key=lambda e: e[0])
    world.step(0.2)                                           # let the camera see the rovers
    states = world.measure()
    if hasattr(decider, 'begin'):
        decider.begin(ctl, instruction)
    ctl.arm()
    t = 0.0
    while t < t_max:
        states = world.measure()
        while events and events[0][0] <= t:
            events.pop(0)[1](world, ctl)
        why = ctl.decision_due(states) if (ctl.armed or not ctl._trig['started']) else []
        if why and decider is not None:
            ob = ctl.observation(states, why, instruction, met)
            rep = decider.decide(ob)
            transcript.append(dict(t=round(t, 1), why=why, observation=ob, reply=_safe(rep)))
            if rep is not None:
                pending.append((t + decision_latency, rep))
        while pending and pending[0][0] <= t:
            rec = ctl.request(pending.pop(0)[1], states, source=getattr(decider, 'name', 'decider'))
            transcript[-1].setdefault('records', []).append(rec)
        cmds = ctl.step(states, dt)
        world.command(cmds)
        world.step(dt)
        t += dt
        working = [n for n in names if n in ctl.s and ctl.route.leg_at(ctl.s[n])['work']
                   and 0.0 <= ctl.s[n] <= ctl.route.length]
        stopped = [n for n in ctl.stop_cause] if ctl.armed else []
        met.update(world.truth(), dt, working, stopped)
        if on_step is not None:
            on_step(t, world, ctl, met)
        if ctl.done or ctl.latched:
            break
    res = met.result()
    res.update(completed=bool(ctl.done), latched=ctl.latched, fleet_stop=ctl.fleet_stop,
               decisions=sum(1 for r in ctl.log if 'applied' in r), rejected=sum(1 for r in ctl.log if 'error' in r),
               held=sum(1 for r in ctl.log if 'held' in r))
    return dict(result=res, log=ctl.log, transcript=transcript, controller=ctl, world=world, metrics=met)


def _safe(o):
    import json
    try:
        json.dumps(o)
        return o
    except Exception:
        return repr(o)
