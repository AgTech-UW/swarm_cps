"""Lay out a demo course on a floor of a given size: staging lane, wall with a
gate, two working passes joined by a headland turn, a rock and a named strip.

The crew is staged in single file along the west side and turns east through
the gate. Staging along the short west side instead of in line with the passes
is what makes three rovers fit on a floor a little over four metres long.

    y = H +-------+--------------------------------+
          |   v   |          north strip           |
          |   v   |   <-----  pass 2  ----------.  |
          |   |   |                             |  |   headland turn
          |   '---gate--->  pass 1  (rock) -----'  |
          |       |                                |
    y = 0 +-------+--------------------------------+
        x = 0    wall                            x = W

The named strip lies along pass 2, so the route AS PROGRAMMED runs through it.
A decider that does nothing about the operator's sentence spends the whole of
pass 2 inside the strip; one that moves the passes away does not. The choice
has to be made on pass 1, which is what makes it a decision and not a reflex.
"""


def make_layout(width, height, n_rovers=3, spacing=0.45, turn_radius=0.55, body_radius=0.14,
                edge_margin=0.20, gate_width=0.80, rock_radius=0.10, strip='north'):
    """Returns (config dict in arena.yaml form, notes). Raises ValueError with
    a plain reason if the floor is too small. strip='north' puts pass 2 and the
    named strip on the north side (crew staged in the north-west corner,
    heading south); 'south' is the mirror image."""
    W, H, R, d = float(width), float(height), float(turn_radius), float(spacing)
    n = max(1, int(n_rovers))
    notes = []
    # Designed with the strip on the low-y side, then mirrored if it should be north.
    col = (n - 1) * d                                   # length of the staged column
    xs = edge_margin + 0.30                             # staging lane
    ys = edge_margin + col + 0.05                       # leader's start
    y_turn = ys + 0.20
    yg = y_turn + R                                     # gate and pass 1
    need = 2 * R + edge_margin + body_radius + 0.06     # room for pass 2 beyond pass 1
    if yg < need <= H - edge_margin - body_radius - 0.25:
        ys, y_turn, yg = ys + (need - yg), y_turn + (need - yg), need
    xw = xs + R + 0.20                                  # wall
    xe = W - edge_margin - R - 0.05                     # east end of the passes
    x1 = xw + 0.25                                      # pass 1 starts just inside the gate
    if xe - x1 < 0.9:
        raise ValueError('floor is %.1f m long; with a %.2f m turn it leaves a %.1f m pass. Need about %.1f m '
                         'or a smaller turn_radius.' % (W, R, max(0.0, xe - x1), x1 + 0.9 + R + edge_margin + 0.05))
    if yg > H - edge_margin - body_radius - 0.25:
        raise ValueError('floor is %.1f m wide; %d rovers staged in single file need about %.1f m. Use fewer '
                         'rovers or a smaller spacing.' % (H, n, yg + edge_margin + body_radius + 0.25))
    two = yg - 2 * R >= edge_margin + body_radius + 0.05
    legs = [dict(name='yard', to=[xs, y_turn, 90]),
            dict(name='gate', to=[x1, yg, 0]),
            dict(name='pass1', to=[xe, yg, 0], work=True)]
    if two:
        y2 = yg - 2 * R
        legs.append(dict(name='headland', to=[xe, y2, 180]))
        legs.append(dict(name='pass2', to=[x1 + 0.15, y2, 180], work=True))
        zone = dict(x=[xw + 0.3, W], y=[0.0, y2 + 0.22])
        rock = dict(name='rock', x=x1 + 0.70 * (xe - x1), y=yg + 0.05, r=rock_radius)
    else:
        notes.append('no room for a second pass: the mission is one pass and stop, and the strip lies along it')
        zone = dict(x=[x1 + 0.9, W], y=[0.0, yg + 0.22])
        rock = dict(name='rock', x=x1 + 0.45, y=yg + 0.05, r=rock_radius)
    walls = [dict(name='gate post A', a=[xw, 0.0], b=[xw, yg - gate_width / 2]),
             dict(name='gate post B', a=[xw, yg + gate_width / 2], b=[xw, H])]
    start = [xs, ys, 90]
    north = (strip == 'north')

    def my(y):
        return H - y if north else y

    def mh(h):
        return -h if north else h

    def r2(v):
        return round(float(v), 2) + 0.0
    zy = sorted([my(zone['y'][0]), my(zone['y'][1])])
    cfg = dict(
        bounds=dict(x=[0.0, r2(W)], y=[0.0, r2(H)]),
        edge_margin=edge_margin,
        compass=dict(plus_x='east', plus_y='north'),
        walls=[dict(name=('the %s gate post' % (('north' if my(w['a'][1] + w['b'][1]) > my(2 * yg) else 'south'))),
                    **{'from': [r2(w['a'][0]), r2(my(w['a'][1]))], 'to': [r2(w['b'][0]), r2(my(w['b'][1]))]}) for w in walls],
        obstacles=[dict(name='rock', x=r2(rock['x']), y=r2(my(rock['y'])), r=rock_radius)],
        zones=[dict(name='%s strip' % ('north' if north else 'south'), x=[r2(zone['x'][0]), r2(zone['x'][1])],
                    y=[r2(zy[0]), r2(zy[1])])],
        mission=dict(turn_radius=R, swath=r2(min(0.30, d)), lead_in=r2(col + 1.0),
                     start=[r2(start[0]), r2(my(start[1])), mh(start[2])],
                     legs=[dict(name=lg['name'], to=[r2(lg['to'][0]), r2(my(lg['to'][1])), mh(lg['to'][2])],
                                **({'work': True} if lg.get('work') else {})) for lg in legs]),
    )
    tail_clear = col + 0.25                             # leader distance past the wall when the tail is through
    open_len = (xe - xw) - tail_clear - 0.7
    if open_len < 0.3:
        notes.append('the passes are short for %d rovers: a wide formation has about %.1f m to open before the turn, '
                     'so expect single file most of the way. The offset and speed choices still show; for formation '
                     'changes use 2 rovers, a smaller spacing, or a longer floor.' % (n, max(0.0, open_len)))
    return cfg, notes


def make_simple_layout(width, height, n_rovers=3, spacing=0.45, turn_radius=0.55, body_radius=0.14,
                       edge_margin=0.20, laps=2, rock=False):
    """Stage 1 course: an empty floor and a racetrack. No wall, no gate, no
    strip; a rock only if asked for. The crew starts in single file on the
    south straight, drives `laps` laps anticlockwise and stops where it began.

        y = H +----------------------------------------+
              |   .----<----- north straight -----.    |
              |   |                               |    |
              |   '-->-- o o O --- south straight-'    |
        y = 0 +----------------------------------------+

    Returns (config, notes); raises ValueError if the floor is too small."""
    W, H, R, d = float(width), float(height), float(turn_radius), float(spacing)
    n = max(1, int(n_rovers))
    col = (n - 1) * d
    clear = edge_margin + body_radius + 0.05
    if H < 2 * R + 2 * clear:
        raise ValueError('floor is %.1f m wide; a loop with %.2f m turns needs %.1f m. Use a smaller turn_radius '
                         '(not below the rover\'s own 0.35 m).' % (H, R, 2 * R + 2 * clear))
    y1 = max(clear, H / 2 - R)
    y2 = y1 + 2 * R
    xw = edge_margin + R + 0.05                         # west end of the straights
    xe = W - edge_margin - R - 0.05                     # east end
    xs = max(edge_margin + col + 0.10, xw + 0.20)       # leader's start: room for the column behind it
    if xe - xs < 0.6:
        raise ValueError('floor is %.1f m long; %d rovers in single file plus two %.2f m turns need about %.1f m. '
                         'Use fewer rovers or a smaller spacing.' % (W, n, R, xs + 0.6 + R + edge_margin + 0.05))

    def r2(v):
        return round(float(v), 2) + 0.0
    legs = []
    for k in range(1, max(1, int(laps)) + 1):
        legs += [dict(name='lap %d south straight' % k, to=[r2(xe), r2(y1), 0], work=True),
                 dict(name='lap %d east turn' % k, to=[r2(xe), r2(y2), 180]),
                 dict(name='lap %d north straight' % k, to=[r2(xw), r2(y2), 180], work=True),
                 dict(name='lap %d west turn' % k, to=[r2(xw), r2(y1), 0])]
    legs.append(dict(name='finish', to=[r2(xs), r2(y1), 0]))
    cfg = dict(bounds=dict(x=[0.0, r2(W)], y=[0.0, r2(H)]), edge_margin=edge_margin,
               compass=dict(plus_x='east', plus_y='north'), walls=[], zones=[], obstacles=[],
               mission=dict(turn_radius=R, swath=r2(min(0.30, d)), lead_in=r2(col + 1.0),
                            start=[r2(xs), r2(y1), 0], legs=legs))
    if rock:
        cfg['obstacles'].append(dict(name='rock', x=r2(xw + 0.5 * (xe - xw)), y=r2(y2 + 0.05), r=0.10))
    return cfg, []


def make_duel_layout(width, height, n_rovers=2, kind='head-on', spacing=0.45, turn_radius=0.55, body_radius=0.14,
                     edge_margin=0.20):
    """Collision-course course: an empty floor, the crew driving east along the
    middle, and the LAST rover in fleet.yaml on a route of its own that meets
    them. kind = 'head-on' (same line, opposite directions) or 'crossing'
    (from the south, timed to arrive at the crossing point together). With
    two rovers it is one against one; with three, a column of two meets one.
    Returns (config, notes)."""
    W, H, d = float(width), float(height), float(spacing)
    n = max(2, int(n_rovers))
    col = (n - 2) * d                                   # followers behind the crew's leader
    yc = H / 2
    xs = edge_margin + col + 0.10
    xe = W - edge_margin - 0.30
    if xe - xs < 2.0:
        raise ValueError('floor is %.1f m long; the rovers need about %.1f m to meet and get past each other'
                         % (W, xs + 2.0 + edge_margin + 0.30))

    def r2(v):
        return round(float(v), 2) + 0.0
    notes = []
    if kind == 'crossing':
        ys = edge_margin + 0.10
        run = yc - ys                                   # solo's distance to the crossing point
        xc = xs + run                                   # so the leader is the same distance from it
        if xc > xe - 0.6 or H - edge_margin - 0.30 - yc < 0.5:
            raise ValueError('floor is too small for a crossing; use --duel head-on')
        solo = dict(robot='last', start=[r2(xc), r2(ys), 90], legs=[dict(name='crossing', to=[r2(xc), r2(H - edge_margin - 0.30), 90])])
        notes.append('both rovers are %.2f m from the crossing point at the start, so at equal speed they arrive together' % run)
    else:
        solo = dict(robot='last', start=[r2(W - edge_margin - 0.10), r2(yc), 180], legs=[dict(name='head-on', to=[r2(edge_margin + 0.30), r2(yc), 180])])
    cfg = dict(bounds=dict(x=[0.0, r2(W)], y=[0.0, r2(H)]), edge_margin=edge_margin,
               compass=dict(plus_x='east', plus_y='north'), walls=[], zones=[], obstacles=[],
               mission=dict(turn_radius=float(turn_radius), swath=r2(min(0.30, d)), lead_in=r2(col + 1.0),
                            start=[r2(xs), r2(yc), 0], legs=[dict(name='run', to=[r2(xe), r2(yc), 0], work=True)],
                            solo=[solo]))
    return cfg, notes


def dump_yaml(cfg, header=''):
    """arena.yaml text with the comments a person needs."""
    def pt(p):
        return '[%s]' % ', '.join('%g' % v for v in p)
    L = []
    if header:
        L += ['# ' + h for h in header.splitlines()] + ['']
    L.append('bounds: {x: %s, y: %s}   # the floor the camera sees and rovers may use (m)' % (pt(cfg['bounds']['x']), pt(cfg['bounds']['y'])))
    L.append('edge_margin: %g                        # routes keep rover centres this far inside' % cfg['edge_margin'])
    L.append('compass: {plus_x: %s, plus_y: %s}' % (cfg['compass']['plus_x'], cfg['compass']['plus_y']))
    L.append('')
    L.append('walls:                                   # lines on the floor; the gap between two is the gate')
    for w in cfg['walls']:
        L.append('  - {name: %s, from: %s, to: %s}' % (w['name'], pt(w['from']), pt(w['to'])))
    if not cfg['walls']:
        L[-1] = 'walls: []'
    L.append('')
    L.append('obstacles:                               # free-standing things the reflex steers round')
    for o in cfg['obstacles']:
        L.append('  - {name: %s, x: %g, y: %g, r: %g}' % (o['name'], o['x'], o['y'], o['r']))
    if not cfg['obstacles']:
        L[-1] = 'obstacles: []                            # add {name: rock, x: , y: , r: 0.1} to try the reflex'
    L.append('')
    L.append('zones:                                   # named floor areas; time spent in each is scored')
    for z in cfg['zones']:
        L.append('  - {name: %s, x: %s, y: %s}' % (z['name'], pt(z['x']), pt(z['y'])))
    if not cfg['zones']:
        L[-1] = 'zones: []'
    m = cfg['mission']
    L += ['', 'mission:',
          '  turn_radius: %g                      # radius of the route\'s turns (the rover\'s own limit is 0.35)' % m['turn_radius'],
          '  swath: %g                             # width each rover "works", for the coverage score' % m['swath'],
          '  lead_in: %g' % m['lead_in'],
          '  start: %s                  # leader start: x, y, heading in degrees (0 = east, 90 = north)' % pt(m['start']),
          '  legs:']
    for lg in m['legs']:
        L.append('    - {name: %s, to: %s%s}' % (lg['name'], pt(lg['to']), ', work: true' if lg.get('work') else ''))
    if m.get('solo'):
        L.append('  solo:                                  # rovers that drive a route of their own ("last" = last in fleet.yaml)')
        for sd in m['solo']:
            L.append('    - robot: %s' % sd['robot'])
            L.append('      start: %s' % pt(sd['start']))
            L.append('      legs:')
            for lg in sd['legs']:
                L.append('        - {name: %s, to: %s}' % (lg['name'], pt(lg['to'])))
    return '\n'.join(L) + '\n'
