"""Rehearse the mission with no hardware and no ROS.

    simulate --check                         # does my arena.yaml work? where do the rovers and the tape go?
    simulate --decider rule --gif run.gif
    simulate --compare --instruction "Neighbour's got cattle along the north fence. Keep the machines well back from that side."
    simulate --decider foreman --model qwen2.5:7b --instruction "..." --gif foreman.gif
    simulate --decider mock --instruction "..."     # offline stand-in for the model (keyword matching)

Deciders: column, schedule, rule, foreman (a real model through Ollama), mock.
"""
import argparse
import json
import sys

from ..core.controller import SwarmController
from ..core.deciders import FixedColumn, Rule, Schedule
from ..core.foreman import Foreman, MockModel, OllamaClient, OpenAIClient
from ..core.sim import run_mission
from ..core.viz import TopDown, write_gif
from .common import cv2, say, settings, verdict


def make_decider(name, args):
    if name == 'column':
        return FixedColumn()
    if name == 'schedule':
        return Schedule()
    if name == 'rule':
        return Rule()
    if name == 'mock':
        f = Foreman(MockModel())
        f.name = 'mock'
        return f
    if name == 'foreman':
        client = OpenAIClient(args.model, args.host) if args.backend == 'openai' else OllamaClient(args.model, args.host)
        ok, msg = client.ping()
        say(('  ' if ok else '  ! ') + msg)
        if not ok:
            sys.exit(2)
        return Foreman(client, votes=args.votes, verbose=True)
    raise SystemExit('unknown decider %r' % name)


def run(st, name, args, gif=None, png=None):
    arena = st.arena()
    dec = make_decider(name, args)
    td = TopDown(arena)
    frames = []
    box = {}

    def on_step(t, world, ctl, met):
        poses = {n: tuple(world.state[n][:3]) for n in world.names}
        td.track(poses)
        box['last'] = (world, ctl, met)
        if gif and int(t * 30) % 12 == 0:
            frames.append(td.draw(ctl, poses, met, title='%s  t = %.0f s' % (name, t), instruction=args.instruction,
                                  thought=getattr(dec, 'last_thought', '')))
    latency = args.latency if name in ('foreman', 'mock') else 0.0
    out = run_mission(arena, st.names, dec, instruction=args.instruction, params=st.params(), on_step=on_step,
                      decision_latency=latency, t_max=args.t_max)
    world, ctl, met = box['last']
    final = td.draw(ctl, {n: tuple(world.state[n][:3]) for n in world.names}, met, title='%s: finished' % name,
                    instruction=args.instruction, thought=getattr(dec, 'last_thought', ''))
    if png:
        cv2.imwrite(png, final)
        say('  picture: %s' % png)
    if gif:
        frames.append(final)
        if write_gif(frames, gif, fps=8):
            say('  animation: %s' % gif)
        else:
            say('  (Pillow not installed: no GIF; use --png)')
    return out


def cmd_check(st, args):
    arena = st.arena()
    p = st.params()
    ctl = SwarmController(arena, st.names, p)
    ok = True
    probs = arena.validate(ctl.route, p.body_radius, p.min_turn_radius)
    for pr in probs:
        ok &= verdict(False, pr)
    ok &= verdict(not probs, 'route is on the floor and clear of the walls (%.1f m long)' % ctl.route.length)
    ok &= verdict(p.spacing > p.protected, 'slot spacing %.2f m is above the separation range %.2f m' % (p.spacing, p.protected))
    say('')
    say('Put the rovers down here (tape an X at each):')
    for n, (x, y, th) in ctl.start_poses().items():
        inside = arena.inside(x, y, 0.12)
        ok &= verdict(inside, '%-8s x = %.2f  y = %.2f  facing %s%s' % (n, x, y, arena.compass(th), '' if inside else
                      '   <- outside the floor: use fewer rovers, smaller spacing, or move mission.start'))
    say('')
    say('Tape on the floor:')
    for w in arena.walls:
        say('  wall  %-22s from (%.2f, %.2f) to (%.2f, %.2f)' % (w['name'], w['a'][0], w['a'][1], w['b'][0], w['b'][1]))
    for o in arena.obstacles:
        say('  %-5s centre (%.2f, %.2f), %.0f cm across (a tin, a bottle, a small box)' % (o['name'], o['x'], o['y'], 200 * o['r']))
    for z in arena.zones:
        say('  zone  %-22s x %.2f to %.2f, y %.2f to %.2f' % (z['name'], z['x0'], z['x1'], z['y0'], z['y1']))
    say('')
    say('Dry run in single file, then with the obvious schedule:')
    for name in ('column', 'schedule'):
        r = run(st, name, args, png='/tmp/swarm_check_%s.png' % name)['result']
        ok &= verdict(r['completed'] and r['collisions'] == 0, '%-8s %s in %.0f s, %d contacts, closest approach %.2f m, field worked %.0f %%'
                      % (name, 'finished' if r['completed'] else 'DID NOT FINISH (%s)' % (r['latched'] or r['fleet_stop'] or 'stuck'),
                         r['time_s'], r['collisions'], r['min_clearance_m'] or 0, r['field_worked_pct']))
    say('')
    say('RESULT: %s' % ('this layout works in simulation' if ok else 'fix the FAIL lines (edit arena.yaml or fleet.yaml)'))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog='simulate', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config-dir', dest='config_dir', default=None)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--compare', action='store_true', help='run rule and the supervisor on the same instruction')
    ap.add_argument('--decider', default='rule')
    ap.add_argument('--instruction', default='')
    ap.add_argument('--model', default='qwen2.5:7b')
    ap.add_argument('--host', default='http://127.0.0.1:11434')
    ap.add_argument('--backend', default='ollama', choices=['ollama', 'openai'])
    ap.add_argument('--votes', type=int, default=1)
    ap.add_argument('--latency', type=float, default=2.0, help='seconds each model reply is delayed while rovers keep moving')
    ap.add_argument('--gif', default=None)
    ap.add_argument('--png', default=None)
    ap.add_argument('--t-max', dest='t_max', type=float, default=300.0)
    ap.add_argument('--transcript', default=None, help='save observations and replies to this JSON file')
    args = ap.parse_args(argv)
    st = settings(args)
    if args.check:
        return cmd_check(st, args)
    names = ['rule', args.decider if args.decider not in ('rule', 'column', 'schedule') else 'mock'] if args.compare else [args.decider]
    rows = []
    for name in names:
        say('running %s ...' % name)
        g = args.gif.replace('.gif', '_%s.gif' % name) if (args.gif and args.compare) else args.gif
        pn = args.png.replace('.png', '_%s.png' % name) if (args.png and args.compare) else args.png
        out = run(st, name, args, gif=g, png=pn)
        rows.append((name, out['result']))
        if args.transcript:
            path = args.transcript.replace('.json', '_%s.json' % name) if args.compare else args.transcript
            with open(path, 'w') as f:
                json.dump(dict(result=out['result'], log=out['log'], transcript=out['transcript']), f, indent=1)
            say('  transcript: %s' % path)
    say('')
    if args.instruction:
        say('instruction: "%s"' % args.instruction)
    zones = sorted({z for _, r in rows for z in r['zone_machine_seconds']})
    say('%-9s %-8s %6s %8s %9s %8s %s' % ('decider', 'finished', 'time', 'contacts', 'closest', 'worked', '  '.join('s in ' + z for z in zones)))
    for name, r in rows:
        say('%-9s %-8s %5.0fs %8d %8.2fm %7.0f%% %s' % (name, 'yes' if r['completed'] else 'NO', r['time_s'], r['collisions'],
                                                      r['min_clearance_m'] or 0, r['field_worked_pct'],
                                                      '  '.join('%10.0f' % r['zone_machine_seconds'][z] for z in zones)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
