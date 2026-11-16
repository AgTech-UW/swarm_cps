"""Formations: where each follower's slot is relative to the leader.

A slot is (along, lat) in the leader's track frame: along < 0 is behind the
leader along the route it has driven, lat > 0 is to the left of travel. All
slots are in the track frame here (the Foreman paper's column formations
follow the leader's track; its line and vee are fixed to the leader's heading
and "swing sideways in turns"). On a small floor the track frame is the only
one that goes round a headland turn, and the fit check keeps wide formations
out of turns anyway, so the two frames give the same behaviour where a wide
formation is allowed.

The names and meanings match shepherd/farmagents.py FARM_PRESETS so the same
supervisor prompt and replies work here.
"""
import itertools
import math

PRESETS = ('line', 'lineLeft', 'lineRight', 'vee', 'column2', 'column')

DESCRIPTIONS = {
    'line':      'working line: followers abreast either side of the leader, one rank back',
    'lineLeft':  'half line folded LEFT: followers on the leader\'s left only',
    'lineRight': 'half line folded RIGHT: followers on the leader\'s right only',
    'vee':       'vee, 45 degrees',
    'column2':   'column two abreast, following the leader\'s track',
    'column':    'single file, following the leader\'s track',
}

_ALIASES = {'singlefile': 'column', 'single': 'column', 'file': 'column', 'single_file': 'column',
            'lineleft': 'lineLeft', 'lineright': 'lineRight', 'line_left': 'lineLeft', 'line_right': 'lineRight',
            'left': 'lineLeft', 'right': 'lineRight', 'column_2': 'column2', 'twoabreast': 'column2',
            'abreast': 'line', 'v': 'vee', 'wedge': 'vee'}


def preset_slots(name, n, d):
    """Slots for `n` followers at spacing `d` metres."""
    out = []
    if name == 'column':
        out = [(-(i + 1) * d, 0.0) for i in range(n)]
    elif name == 'column2':
        for i in range(n):
            row, col = divmod(i, 2)
            m = min(2, n - row * 2)
            out.append((-(row + 1) * d, (col - (m - 1) / 2.0) * d if m > 1 else 0.0))
        if n == 1:
            out = [(-d, 0.0)]
    elif name == 'vee':
        c = math.cos(math.radians(45))
        for i in range(n):
            arm = (i // 2 + 1) * d
            side = 1.0 if i % 2 == 0 else -1.0
            out.append((-arm * c, side * arm * c))
    elif name == 'line':
        for i in range(n):
            k = i // 2 + 1
            side = 1.0 if i % 2 == 0 else -1.0
            out.append((-d, side * k * d))
    elif name in ('lineLeft', 'lineRight'):
        sg = 1.0 if name == 'lineLeft' else -1.0
        lanes = max(1, (n + 1) // 2)                 # two ranks, as in the paper's folded half line
        for i in range(n):
            row, col = divmod(i, lanes)
            out.append((-(row + 1) * d, sg * (col + 1) * d))
    else:
        raise ValueError('unknown formation %r' % (name,))
    return out


def lanes_of(slots):
    """Sorted distinct lateral positions of all machines (leader at 0)."""
    return sorted(set([0.0] + [round(l, 3) for _, l in slots]))


def length_of(slots):
    return max([0.0] + [-a for a, _ in slots])


def half_width(slots):
    return max([0.0] + [abs(l) for _, l in slots])


def canonical_name(text):
    """Any capitalisation or spelling of a preset name -> the preset name, or None."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    for p in PRESETS:
        if t.lower() == p.lower():
            return p
    key = ''.join(ch for ch in t.lower() if ch.isalnum() or ch == '_')
    key2 = key.replace('_', '')
    return _ALIASES.get(key) or _ALIASES.get(key2)


def check_custom(slots, n, min_gap, max_lat, max_along):
    """The rules the prompt states for a custom formation, enforced: one slot
    per follower, all behind the leader, every pair of machines (leader
    included) more than min_gap apart, nothing absurdly far away. Returns a
    list of (along, lat) or raises ValueError saying which rule broke."""
    if not isinstance(slots, (list, tuple)):
        raise ValueError('custom formation needs "slots": [[along, lane], ...]')
    pts = []
    for p in slots:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError('each slot must be [along, lane]')
        try:
            a, l = float(p[0]), float(p[1])
        except Exception:
            raise ValueError('slot values must be numbers')
        if not (math.isfinite(a) and math.isfinite(l)):
            raise ValueError('slot values must be finite')
        pts.append((a, l))
    if len(pts) != n:
        raise ValueError('custom formation has %d slots for %d followers' % (len(pts), n))
    for a, l in pts:
        if a >= 0:
            raise ValueError('slot [%g, %g] is not behind the leader (along must be < 0)' % (a, l))
        if abs(l) > max_lat or -a > max_along:
            raise ValueError('slot [%g, %g] is too far from the leader' % (a, l))
    allp = [(0.0, 0.0)] + pts
    for i in range(len(allp)):
        for j in range(i + 1, len(allp)):
            if math.hypot(allp[i][0] - allp[j][0], allp[i][1] - allp[j][1]) <= min_gap:
                raise ValueError('two machines would be %.2f m apart or less (need more than %.2f m)'
                                 % (math.hypot(allp[i][0] - allp[j][0], allp[i][1] - allp[j][1]), min_gap))
    return pts


def resolve(f, n, d, min_gap, max_lat=5.0, max_along=20.0):
    """A formation as a supervisor might write it -> (name, slots).

    Read leniently, checked strictly (the paper's "interface 2"): a preset may
    be given as "column", "Column", {"shape": "column"}, {"name": ...} or
    {"formation": ...}; a custom formation must pass check_custom."""
    if isinstance(f, str):
        name = canonical_name(f)
        if name is None:
            raise ValueError('unknown formation %r; use one of %s' % (f, ', '.join(PRESETS)))
        return name, preset_slots(name, n, d)
    if isinstance(f, dict):
        for key in ('shape', 'name', 'formation', 'preset', 'type'):
            v = f.get(key)
            if isinstance(v, str) and v.strip().lower() != 'custom':
                name = canonical_name(v)
                if name is None:
                    raise ValueError('unknown formation %r; use one of %s' % (v, ', '.join(PRESETS)))
                return name, preset_slots(name, n, d)
        if 'slots' in f:
            return 'custom', check_custom(f['slots'], n, min_gap, max_lat, max_along)
        raise ValueError('formation object has neither a preset name nor slots')
    raise ValueError('formation must be a name or an object, got %s' % type(f).__name__)


def assign(current, slots):
    """Which follower takes which slot: minimum total squared travel.
    current: [(along, lat)] of each follower now, in the leader's track frame.
    Returns perm with perm[i] = index of the slot follower i takes. Assigning
    by index instead makes followers drive through each other on a reshuffle."""
    n = len(current)
    if n == 0:
        return []
    if n != len(slots):
        raise ValueError('assign: %d followers, %d slots' % (n, len(slots)))

    def cost(i, j):
        return (current[i][0] - slots[j][0]) ** 2 + (current[i][1] - slots[j][1]) ** 2
    if n <= 7:
        best = None
        for perm in itertools.permutations(range(n)):
            c = sum(cost(i, perm[i]) for i in range(n))
            if best is None or c < best[0] - 1e-12:
                best = (c, perm)
        return list(best[1])
    free = set(range(n))                             # greedy for big crews
    perm = [None] * n
    for i in sorted(range(n), key=lambda i: -min(cost(i, j) for j in range(n))):
        j = min(free, key=lambda j: cost(i, j))
        perm[i] = j
        free.discard(j)
    return perm


def table(n, d, swath):
    """The formation table shown to the supervisor."""
    rows = []
    for name in PRESETS:
        sl = preset_slots(name, n, d)
        ln = lanes_of(sl)
        rows.append('  %-9s %-66s working width %.2f m, %d lane%s, length %.2f m'
                    % (name, DESCRIPTIONS[name], (ln[-1] - ln[0]) + swath, len(ln), '' if len(ln) == 1 else 's',
                       length_of(sl)))
    return '\n'.join(rows)
