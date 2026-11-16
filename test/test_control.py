"""The crew controller, the safety layers and the deciders, run against the
simulated rovers (actuator lag, turning limit, noisy late camera)."""
import math
import os

import numpy as np
import pytest

from swarm_cps.core import formation as F
from swarm_cps.core.arena import Arena
from swarm_cps.core.controller import Params, RobotState, SwarmController, limit_command
from swarm_cps.core.deciders import FixedColumn, Rule, Schedule
from swarm_cps.core.foreman import Foreman, MockModel, build_system, parse_json
from swarm_cps.core.layout import make_duel_layout, make_layout, make_simple_layout
from swarm_cps.core.path import Route, _integrate, dubins_shortest, wrap
from swarm_cps.core.sim import SimWorld, run_mission

NAMES = ['rover1', 'rover2', 'rover3']
CATTLE = "Neighbour's got cattle along the north fence. Keep the machines well back from that side."


def arena(w=5.5, h=3.5, n=3):
    return Arena(make_layout(w, h, n)[0])


def placed(ctl):
    """States with every rover exactly on its start mark."""
    return {n: RobotState(*p) for n, p in ctl.start_poses().items()}


# ---------------------------------------------------------------- limits and paths
def test_limits_keep_curvature_and_never_turn_in_place():
    p = Params()
    v, w = limit_command(0.5, 1.0, p)                           # too fast: both scaled together
    assert v == p.v_max and abs(w / v - 1.0 / 0.5) < 1e-9
    v, w = limit_command(0.1, 5.0, p)                           # tighter than the chassis can turn
    assert abs(abs(w) - 0.1 / p.min_turn_radius) < 1e-9
    assert limit_command(0.0, 0.8, p) == (0.0, 0.0)
    assert limit_command(-0.2, 0.3, p) == (0.0, 0.0)
    assert limit_command(float('nan'), 0.0, p) == (0.0, 0.0)


def test_dubins_reaches_every_goal():
    rng = np.random.default_rng(0)
    for _ in range(1500):
        q0 = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
        q1 = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
        R = rng.uniform(0.3, 1.0)
        e = _integrate(q0, dubins_shortest(q0, q1, R), R)
        assert math.hypot(e[0] - q1[0], e[1] - q1[1]) < 1e-6 and abs(wrap(e[2] - q1[2])) < 1e-6


def test_route_offset_moves_passes_but_not_the_gate():
    ar = arena()
    r = ar.make_route()
    gate = next(lg for lg in r.legs if lg['name'] == 'gate')
    p1 = next(lg for lg in r.legs if lg['name'] == 'pass1')
    g0 = r.at((gate['s0'] + gate['s1']) / 2 - 0.3)
    m0 = r.at((p1['s0'] + p1['s1']) / 2 + 0.5)
    r.set_offset(-0.4)
    g1 = r.at((gate['s0'] + gate['s1']) / 2 - 0.3)
    m1 = r.at((p1['s0'] + p1['s1']) / 2 + 0.5)
    assert abs(g1[1] - g0[1]) < 1e-9                            # the gate stays where it is
    assert abs((m1[1] - m0[1]) + 0.4) < 1e-6                    # the pass moved 0.4 m south
    assert np.abs(r.kappa).max() < 1.0 / 0.35                   # and the shifted route is still drivable


# ---------------------------------------------------------------- formations and the reply interface
def test_presets_keep_machines_apart():
    for n in (1, 2, 3, 4, 5):
        for name in F.PRESETS:
            sl = [(0.0, 0.0)] + F.preset_slots(name, n, 0.45)
            assert len(sl) == n + 1
            d = min(math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(sl) for b in sl[i + 1:])
            assert d >= 0.44, (name, n, d)


def test_replies_are_read_leniently_and_checked_strictly():
    ctl = SwarmController(arena(), NAMES)
    st = placed(ctl)
    for ok in ('column', 'Column', 'LINELEFT', {'shape': 'column2'}, {'name': 'Vee'}, 'single file',
               {'formation': 'lineRight', 'speed': 0.1}):
        rec = ctl.request(dict(formation=ok, speed=0.12, offset=0), st)
        assert 'error' not in rec, (ok, rec)
    before = (ctl.formation, ctl.speed)
    bad = [dict(formation='diamond'), dict(speed=0.1), 'column', None, dict(formation=dict(shape='custom', slots=[[-0.5, 0]])),
           dict(formation=dict(shape='custom', slots=[[0.5, 0.3], [-0.5, 0]])),         # a slot ahead of the leader
           dict(formation=dict(shape='custom', slots=[[-0.2, 0.0], [-0.4, 0.0]])),      # machines 0.2 m apart
           dict(formation='column', speed='fast'), dict(formation='column', offset=float('nan'))]
    for b in bad:
        rec = ctl.request(b, st)
        assert 'error' in rec, b
        assert (ctl.formation, ctl.speed) == before             # an unusable reply changes nothing
    good = ctl.request(dict(formation=dict(shape='custom', slots=[[-0.5, 0.3], [-0.5, -0.3]]), speed=9, offset=0), st)
    assert 'error' not in good and ctl.speed == ctl.p.speed_range[1]     # speed clamped, not rejected
    assert parse_json('Sure! {"formation": "column", "speed": 0.1} hope that helps')['formation'] == 'column'
    assert parse_json('no json here') is None


def test_safety_filter_holds_a_wide_formation_in_the_yard_and_releases_it_in_the_field():
    ar = arena()
    out = run_mission(ar, NAMES, Schedule())
    log = out['log']
    held = [r for r in log if r.get('held') == 'line']
    released = [r for r in log if r.get('released') == 'line']
    assert held and released
    assert held[0]['applied']['formation'] == 'column'          # asked for line at the gate, stayed in single file
    assert released[0]['t'] > held[0]['t']
    assert out['result']['collisions'] == 0 and out['result']['completed']


# ---------------------------------------------------------------- whole missions
@pytest.mark.parametrize('floor', [(4.2, 3.2, 3), (4.2, 3.2, 2), (5.5, 3.5, 3), (6.0, 4.0, 4)])
@pytest.mark.parametrize('decider', ['column', 'schedule', 'rule'])
def test_missions_finish_without_contact(floor, decider):
    w, h, n = floor
    ar = arena(w, h, n)
    names = ['rover%d' % (i + 1) for i in range(n)]
    dec = dict(column=FixedColumn, schedule=Schedule, rule=Rule)[decider]()
    out = run_mission(ar, names, dec, instruction='work the field')
    r = out['result']
    assert r['completed'], r
    assert r['collisions'] == 0, r['collision_log']
    assert r['latched'] is None
    assert r['min_clearance_m'] > 0.0


def test_mission_survives_a_chassis_that_is_not_what_the_model_assumes():
    """Wider real turning circle than configured, one slow rover, more camera
    noise and latency, 10 % dropped detections."""
    ar = arena()
    ctl = SwarmController(ar, NAMES)
    world = SimWorld(NAMES, ctl.start_poses(), true_turn_radius=0.45, tau_v=0.4, tau_w=0.35, pos_noise=0.01,
                     yaw_noise=math.radians(2.5), latency=0.18, dropout=0.10, v_scale={'rover3': 0.8}, seed=5)
    out = run_mission(ar, NAMES, Schedule(), sim=world)
    assert out['result']['completed'] and out['result']['collisions'] == 0, out['result']


def test_rovers_put_down_carelessly_still_form_up():
    ar = arena()
    ctl = SwarmController(ar, NAMES)
    poses = ctl.start_poses()
    poses['rover2'] = (poses['rover2'][0] + 0.12, poses['rover2'][1] - 0.05, poses['rover2'][2] + 0.5)
    poses['rover3'] = (poses['rover3'][0] - 0.10, poses['rover3'][1] + 0.04, poses['rover3'][2] - 0.4)
    out = run_mission(ar, NAMES, FixedColumn(), start_poses=poses)
    assert out['result']['completed'] and out['result']['collisions'] == 0, out['result']


# ---------------------------------------------------------------- stops
def test_stale_or_missing_pose_stops_the_whole_fleet():
    ctl = SwarmController(arena(), NAMES)
    st = placed(ctl)
    ctl.arm()
    assert any(v > 0 for v, _ in ctl.step(st, 0.03).values())
    st['rover3'].age = 0.8
    assert all(c == (0.0, 0.0) for c in ctl.step(st, 0.03).values()) and 'stale' in ctl.fleet_stop
    st['rover3'].age = 0.05
    assert any(v > 0 for v, _ in ctl.step(st, 0.03).values())    # fresh again: carries on
    del st['rover2']
    assert all(c == (0.0, 0.0) for c in ctl.step(st, 0.03).values()) and 'no pose' in ctl.fleet_stop
    st['rover2'] = RobotState(float('nan'), 0, 0)
    assert all(c == (0.0, 0.0) for c in ctl.step(st, 0.03).values())


def test_leaving_the_floor_latches_until_reset():
    ctl = SwarmController(arena(), NAMES)
    st = placed(ctl)
    ctl.arm()
    st['rover2'] = RobotState(-0.3, 1.0, 0.0)
    assert all(c == (0.0, 0.0) for c in ctl.step(st, 0.03).values())
    assert ctl.latched and not ctl.armed
    st = placed(ctl)
    assert all(c == (0.0, 0.0) for c in ctl.step(st, 0.03).values())   # back inside, still stopped
    assert not ctl.arm()[0]
    ctl.reset()
    assert ctl.arm()[0]


def test_disarmed_controller_commands_nothing():
    ctl = SwarmController(arena(), NAMES)
    assert all(c == (0.0, 0.0) for c in ctl.step(placed(ctl), 0.03).values())


def test_nobody_drives_into_a_machine_that_is_in_the_way():
    ctl = SwarmController(arena(), NAMES)
    st = placed(ctl)
    x, y, yaw = ctl.start_poses()['rover1']
    st['rover3'] = RobotState(x + 0.30 * math.cos(yaw), y + 0.30 * math.sin(yaw), yaw)   # parked in front of the leader
    ctl.arm()
    cmd = ctl.step(st, 0.03)
    assert cmd['rover1'] == (0.0, 0.0) and 'in the way' in ctl.stop_cause['rover1']


def test_dead_follower_does_not_get_rammed_and_the_leader_waits():
    """rover2's motors die on pass 1. rover3, behind it, must brake rather
    than hit it, and the leader must slow instead of leaving the crew."""
    ar = arena()
    out = run_mission(ar, NAMES, FixedColumn(), t_max=70.0,
                      events=[(28.0, lambda world, ctl: world.frozen.add('rover2'))])
    r = out['result']
    assert r['collisions'] == 0, r['collision_log']
    assert not r['completed']                                   # it does not pretend to finish
    ctl, world = out['controller'], out['world']
    assert world.state['rover3'][3] < 0.01                      # rover3 has stopped behind the dead one
    assert world.state['rover1'][3] <= ctl.p.speed_default * ctl.p.leader_wait[2] + 0.02


def test_camera_losing_one_rover_stops_everyone_and_recovery_resumes():
    ar = arena()
    seen = {}

    def lose(world, ctl):
        world.blind.add('rover2')

    def check(world, ctl):
        seen['stop'] = ctl.fleet_stop
        seen['speed'] = max(world.state[n][3] for n in NAMES)
        world.blind.discard('rover2')
    out = run_mission(ar, NAMES, FixedColumn(), events=[(20.0, lose), (24.0, check)])
    assert 'rover2' in (seen['stop'] or '') and seen['speed'] < 0.01
    assert out['result']['completed'] and out['result']['collisions'] == 0


def test_reflex_takes_the_crew_round_the_rock():
    ar = arena()
    out = run_mission(ar, NAMES, FixedColumn())
    rock = ar.obstacles[0]
    hits = [c for c in out['result']['collision_log'] if c['kind'] == 'obstacle']
    assert not hits
    # the rock sits on the leader's line, so without the reflex this run hits it
    p = Params(reflex_look=0.0)
    out2 = run_mission(ar, NAMES, FixedColumn(), params=p, t_max=120)
    assert any(c['kind'] == 'obstacle' for c in out2['result']['collision_log']) or not out2['result']['completed']
    assert rock['r'] > 0


# ---------------------------------------------------------------- the foreman
def test_prompt_states_the_facts_the_model_needs():
    ctl = SwarmController(arena(), NAMES)
    s = build_system(ctl, CATTLE)
    for needle in (CATTLE, 'north (+) or south (-)', 'north strip', 'pass1 (working pass heading east',
                   'lineRight', 'formationsThatFit', '"offset": <m>'):
        assert needle in s, needle


def test_instruction_nobody_coded_for_changes_what_the_crew_does():
    """The demonstration in one test. Same floor, same rovers, same sentence.
    The hand-written rule has no handler for it and drives its usual lines;
    the supervisor moves the work away from the north strip. (The model here
    is the offline stand-in, so this checks the plumbing from sentence to
    wheels, not a language model's judgement.)"""
    ar = arena()
    rule = run_mission(ar, NAMES, Rule(), instruction=CATTLE)['result']
    fm = Foreman(MockModel())
    sup = run_mission(ar, NAMES, fm, instruction=CATTLE, decision_latency=2.0)
    res = sup['result']
    assert rule['completed'] and res['completed']
    assert rule['collisions'] == 0 and res['collisions'] == 0
    assert rule['zone_machine_seconds']['north strip'] > 5.0
    assert res['zone_machine_seconds']['north strip'] < 1.0
    assert res['field_worked_pct'] > 0.6 * rule['field_worked_pct']   # and it still worked the field
    assert fm.n_bad >= 1                                        # malformed replies happened and were survived
    assert any(isinstance(r.get('asked'), dict) and isinstance(r['asked'].get('formation'), dict)
               for r in sup['log'])                             # a preset written as an object was accepted


def test_a_model_that_only_talks_nonsense_leaves_the_crew_in_single_file():
    class Garbage:
        def chat(self, messages):
            return 'I would be happy to help with your farm!'
    out = run_mission(arena(), NAMES, Foreman(Garbage()), instruction=CATTLE)
    assert out['result']['completed'] and out['result']['collisions'] == 0
    assert out['controller'].formation == 'column'


def test_a_model_that_asks_for_the_impossible_is_overruled():
    class Reckless:
        def chat(self, messages):
            return '{"thought": "wide and fast everywhere", "formation": "line", "speed": 5, "offset": 3}'
    out = run_mission(arena(), NAMES, Foreman(Reckless()), instruction='go')
    r = out['result']
    assert r['completed'] and r['collisions'] == 0, r
    assert r['held'] >= 1
    assert out['controller'].speed <= out['controller'].p.speed_range[1]
    assert abs(out['controller'].offset_target) <= 1.0


def test_a_model_that_goes_silent_after_asking_for_a_wide_formation_is_folded_for_the_turn():
    """The line is in force and the model stops answering. Nobody asks for
    single file before the headland turn; the filter narrows the crew itself
    and gives the line back on the next pass."""
    class Once:
        def __init__(self):
            self.n = 0

        def chat(self, messages):
            self.n += 1
            if self.n <= 3:
                return '{"formation": "line", "speed": 0.13, "offset": 0}'
            raise OSError('connection refused')
    out = run_mission(arena(), NAMES, Foreman(Once()), instruction='go')
    assert out['result']['completed'] and out['result']['collisions'] == 0, out['result']
    narrowed = [r for r in out['log'] if r.get('narrowed_to')]
    released = [r for r in out['log'] if r.get('released') == 'line']
    assert narrowed and narrowed[0]['narrowed_to'] == 'column' and len(released) >= 2


def test_a_model_server_that_is_down_does_not_stop_the_mission():
    class Down:
        def chat(self, messages):
            raise OSError('connection refused')
    out = run_mission(arena(), NAMES, Foreman(Down()), instruction=CATTLE)
    assert out['result']['completed'] and out['result']['collisions'] == 0


def test_layout_refuses_a_floor_that_is_too_small():
    with pytest.raises(ValueError):
        make_layout(2.0, 2.0, 3)
    cfg, notes = make_layout(4.2, 3.2, 3)
    ar = Arena(cfg)
    ctl = SwarmController(ar, NAMES)
    assert ar.validate(ctl.route, ctl.p.body_radius, ctl.p.min_turn_radius) == []
    for n, (x, y, _) in ctl.start_poses().items():
        assert ar.inside(x, y, 0.15), n


def test_route_is_a_route():
    assert isinstance(arena().make_route(), Route)


# ---------------------------------------------------------------- stage 1: just follow the leader
@pytest.mark.parametrize('floor', [(3.6, 2.4, 2), (4.2, 3.2, 3), (5.5, 3.5, 4)])
def test_stage_one_loop_in_single_file(floor):
    """Empty floor, a racetrack, no decider at all: the leader follows the
    route with the vector field, the others follow it and keep their distance."""
    w, h, n = floor
    ar = Arena(make_simple_layout(w, h, n, laps=2)[0])
    names = ['rover%d' % (i + 1) for i in range(n)]
    gaps = []

    def watch(t, world, ctl, met):
        if t > 8.0 and not ctl.done:
            gaps.extend(math.hypot(world.state[a][0] - world.state[b][0], world.state[a][1] - world.state[b][1])
                        for a, b in zip(names, names[1:]))
    out = run_mission(ar, names, None, on_step=watch, t_max=400)
    r = out['result']
    assert r['completed'] and r['collisions'] == 0, r
    assert r['min_clearance_m'] > 0.05
    assert out['controller'].formation == 'column'
    # neighbours stay about one slot spacing apart the whole way round (chords are shorter in the turns)
    assert 0.30 < min(gaps) and max(gaps) < 0.62, (min(gaps), max(gaps))
    # and they end where they started, lined up
    x0, y0, _ = out['controller'].start_poses()[names[0]]
    assert math.hypot(out['world'].state[names[0]][0] - x0, out['world'].state[names[0]][1] - y0) < 0.08


def test_stage_one_with_a_rock_and_a_stalled_rover():
    ar = Arena(make_simple_layout(4.2, 3.2, 3, laps=1, rock=True)[0])
    out = run_mission(ar, NAMES, None)
    assert out['result']['completed'] and out['result']['collisions'] == 0, out['result']
    # one rover's motors cut out for six seconds mid-lap: the one behind waits, nobody touches, the run finishes
    out = run_mission(ar, NAMES, None, t_max=400, events=[(15.0, lambda w, c: w.frozen.add('rover2')),
                                                          (21.0, lambda w, c: w.frozen.discard('rover2'))])
    assert out['result']['completed'] and out['result']['collisions'] == 0, out['result']


# ---------------------------------------------------------------- collision course
def _duel(kind, n=2, floor=(4.2, 3.2), **kw):
    ar = Arena(make_duel_layout(floor[0], floor[1], n, kind)[0])
    names = ['rover%d' % (i + 1) for i in range(n)]
    track = {m: [] for m in names}

    def watch(t, world, ctl, met):
        for m in names:
            track[m].append((world.state[m][0], world.state[m][1]))
    out = run_mission(ar, names, None, on_step=watch, t_max=120, **kw)
    return ar, out, {m: np.array(v) for m, v in track.items()}


def test_two_rovers_head_on_both_keep_right_and_pass():
    ar, out, tr = _duel('head-on')
    r = out['result']
    assert r['completed'] and r['collisions'] == 0, r
    assert r['min_clearance_m'] > 0.05 and r['stopped_machine_seconds'] == 0.0     # they flowed past; nobody had to stop
    yc = ar.start[1]
    assert tr['rover1'][:, 1].min() < yc - 0.12       # east-bound: its right is south
    assert tr['rover2'][:, 1].max() > yc + 0.12       # west-bound: its right is north
    assert tr['rover1'][:, 1].max() < yc + 0.05 and tr['rover2'][:, 1].min() > yc - 0.05   # neither went left
    # and both came back to the line and finished on their marks
    assert abs(tr['rover1'][-1, 1] - yc) < 0.05 and abs(tr['rover2'][-1, 1] - yc) < 0.05


def test_crossing_the_one_with_the_other_on_its_right_gives_way():
    ar, out, tr = _duel('crossing')
    r = out['result']
    assert r['completed'] and r['collisions'] == 0, r
    xc = ar.solo[0]['start'][0]
    assert np.abs(tr['rover2'][:, 0] - xc).max() < 0.05                 # the stand-on rover held its course
    assert tr['rover1'][:, 1].min() < ar.start[1] - 0.15                # the give-way rover went behind it


@pytest.mark.parametrize('case', [('head-on', 3, (4.2, 3.2)), ('crossing', 3, (4.2, 3.2)), ('head-on', 4, (5.5, 3.5))])
def test_a_column_meets_a_lone_rover(case):
    kind, n, floor = case
    ar, out, tr = _duel(kind, n, floor)
    assert out['result']['completed'] and out['result']['collisions'] == 0, out['result']
    assert out['controller'].followers == ['rover%d' % (i + 1) for i in range(1, n - 1)]


def test_collision_course_without_the_passing_rule_ends_in_a_safe_standoff():
    """Turn the right-of-way rule off and the last-resort stop is all that is
    left: the rovers stop nose to nose, untouched, and the run does not finish."""
    for kind in ('head-on', 'crossing'):
        ar, out, tr = _duel(kind, params=Params(avoid_look=0.0))
        assert out['result']['collisions'] == 0, (kind, out['result'])
        assert not out['result']['completed']


def test_collision_course_with_sloppy_hardware():
    ar = Arena(make_duel_layout(4.2, 3.2, 2, 'head-on')[0])
    names = ['rover1', 'rover2']
    for seed in range(4):
        ctl = SwarmController(ar, names)
        world = SimWorld(names, ctl.start_poses(), true_turn_radius=0.45, tau_v=0.4, tau_w=0.35, pos_noise=0.01,
                         yaw_noise=math.radians(2.5), latency=0.18, dropout=0.10, v_scale={'rover2': 0.8}, seed=seed)
        out = run_mission(ar, names, None, sim=world, t_max=120)
        assert out['result']['collisions'] == 0 and out['result']['completed'], (seed, out['result'])


# ---------------------------------------------------------------- a route you draw yourself
def test_a_drawn_route_is_driven(tmp_path, monkeypatch):
    import shutil
    from swarm_cps.settings import Settings
    from swarm_cps.tools import draw_path
    cfg = tmp_path / 'cfg'
    shutil.copytree(os.path.join(os.path.dirname(__file__), '..', 'config'), str(cfg))
    monkeypatch.setenv('SWARM_CONFIG_DIR', str(cfg))
    import yaml
    a = yaml.safe_load(open(str(cfg / 'arena.yaml')))
    a['walls'], a['obstacles'], a['zones'] = [], [], []              # an empty floor to draw on
    yaml.safe_dump(a, open(str(cfg / 'arena.yaml'), 'w'))
    assert draw_path.main(['--points', '1.2,0.8', '3.2,0.8', '3.4,2.2', '1.4,2.4']) == 0     # an S across the floor
    st = Settings()
    out = run_mission(st.arena(), st.names, None, params=st.params())
    assert out['result']['completed'] and out['result']['collisions'] == 0, out['result']
    x, y = out['world'].state['rover1'][:2]
    assert math.hypot(x - 1.4, y - 2.4) < 0.08                       # the leader stopped on the last point
    # points off the floor, or too close to turn between, are refused and nothing is written
    before = open(str(cfg / 'arena.yaml')).read()
    assert draw_path.main(['--points', '1.2,0.8', '5.5,0.8']) == 1
    assert draw_path.main(['--points', '1.5,1.0', '2.4,1.0', '1.5,1.2']) == 1
    assert open(str(cfg / 'arena.yaml')).read() == before
