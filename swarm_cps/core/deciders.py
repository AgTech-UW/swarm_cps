"""Deciders that are not a language model. Every decider chooses the same
three things, formation, leader speed and pass offset, at the same decision
points, and the controller carries them out. These are the comparators of the
Foreman paper, for the A/B half of the demonstration:

  FixedColumn   single file throughout. Safe, gets little done.
  Schedule      the obvious program: single file off the passes, the full
                working line on them. Knows nothing about hazards or words.
  Rule          what an engineer would add by hand: the schedule plus two
                instruction handlers, 'wet' (keep lanes out of zones marked
                kind: wet) and 'rain' (work wide and fast). Rocks are left to
                the reflex: at this scale a rover with its clearance is wider
                than the gap between lanes, so there is no "rock between
                lanes" to plan for. Any other instruction falls through to
                the default behaviour, because nobody wrote a handler for it.
                That gap is the point of the comparison.
"""
from . import formation as F


class FixedColumn:
    name = 'column'

    def __init__(self, speed=None):
        self.speed = speed

    def begin(self, ctl, instruction=''):
        self.ctl = ctl

    def decide(self, ob):
        return dict(formation='column', speed=self.speed or self.ctl.p.speed_default, offset=0.0)


class Schedule:
    name = 'schedule'

    def __init__(self, work='line', fold=0.5):
        self.work, self.fold = work, fold

    def begin(self, ctl, instruction=''):
        self.ctl = ctl

    def _formation(self, ob):
        ps = ob.get('pass')
        if ps is None or not ps.get('working') or ps['metresLeft'] < self.fold:
            return 'column'
        return self.work

    def decide(self, ob):
        return dict(formation=self._formation(ob), speed=self.ctl.p.speed_default, offset=0.0)


class Rule(Schedule):
    """Hand-written rule. It reads the instruction only to pick one of the two
    handlers it has: text containing 'rain' selects the rain handler (work
    wide and fast), anything else the default 'wet' handler (keep lanes out
    of zones marked kind: wet)."""
    name = 'rule'

    def begin(self, ctl, instruction=''):
        self.ctl = ctl
        self.mode = 'rain' if 'rain' in (instruction or '').lower() else 'wet'
        self.current = None

    def _score(self, name, off, ob):
        ctl = self.ctl
        slots = F.preset_slots(name, ctl.n, ctl.p.spacing)
        lanes = F.lanes_of(slots)
        sc = 3.0 * len(lanes) * (1.6 if self.mode == 'rain' else 1.0) - 2.0 * abs(off)
        if self.mode == 'wet':
            ar, r = ctl.arena, ctl.route
            for z in ar.zones:
                if z['kind'] != 'wet':
                    continue
                for l in lanes:
                    x, y, th, _ = r.lane_point(ctl.s_leader + 0.4, l)
                    x += (off - ctl.offset_applied) * r.axis[0]
                    y += (off - ctl.offset_applied) * r.axis[1]
                    if z['x0'] <= x <= z['x1'] and z['y0'] <= y <= z['y1']:
                        sc -= 20.0
        return sc

    def decide(self, ob):
        ctl = self.ctl
        speed = ctl.p.speed_range[1] if self.mode == 'rain' else ctl.p.speed_default
        if self._formation(ob) == 'column':
            self.current = None
            return dict(formation='column', speed=ctl.p.speed_default)
        best, best_sc = ('line', 0.0), -1e18
        for name, (lo, hi) in ob['pass']['allowedOffsetM'].items():
            offs = [k * 0.05 for k in range(int(round(lo / 0.05)), int(round(hi / 0.05)) + 1)]
            for off in sorted(offs, key=abs):                 # the smallest shift that does the job
                s = self._score(name, off, ob)
                if s > best_sc + 1e-9:
                    best, best_sc = (name, off), s
        if self.current is not None and self.current[0] in ob['pass']['allowedOffsetM']:
            lo, hi = ob['pass']['allowedOffsetM'][self.current[0]]
            if lo <= self.current[1] <= hi and self._score(self.current[0], self.current[1], ob) >= best_sc - 0.3:
                best = self.current                           # do not chase small differences
        self.current = best
        return dict(formation=best[0], speed=speed, offset=round(best[1], 2))
