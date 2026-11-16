"""Generate arena.yaml for your floor.

    layout --width 4.2 --height 3.2 --rovers 3            # prints it
    layout --width 4.2 --height 3.2 --rovers 3 --write    # writes it into the config directory

Width is the long side (x, "east"), height the short side (y, "north"), both
measured over the floor the camera can see. Afterwards run  simulate --check.
"""
import argparse
import os
import shutil
import sys

from ..core.layout import dump_yaml, make_layout
from .common import say, settings

HEADER = '''The floor, as the rovers and the supervisor know it.

Units are metres in the frame set by the floor calibration (camera_pose.yaml).
"+x" is called east and "+y" north in everything people and the language
model read; they are names for the two floor axes, not real compass points.

Generated for a %(w)g x %(h)g m floor and %(n)d rovers by
    layout --width %(w)g --height %(h)g --rovers %(n)d
Edit freely, then run:  simulate --check'''


def main(argv=None):
    ap = argparse.ArgumentParser(prog='layout', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config-dir', dest='config_dir', default=None)
    ap.add_argument('--width', type=float, required=True)
    ap.add_argument('--height', type=float, required=True)
    ap.add_argument('--rovers', type=int, default=None)
    ap.add_argument('--spacing', type=float, default=None)
    ap.add_argument('--turn-radius', dest='turn_radius', type=float, default=0.55)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args(argv)
    st = settings(args)
    n = args.rovers or len(st.names)
    spacing = args.spacing or st.params().spacing
    try:
        cfg, notes = make_layout(args.width, args.height, n, spacing=spacing, turn_radius=args.turn_radius,
                                 body_radius=st.params().body_radius)
    except ValueError as e:
        say('DOES NOT FIT: %s' % e)
        return 1
    text = dump_yaml(cfg, HEADER % dict(w=args.width, h=args.height, n=n))
    for nt in notes:
        say('NOTE: ' + nt)
    if args.write:
        path = os.path.join(st.dir, 'arena.yaml')
        if os.path.isfile(path):
            shutil.copy(path, path + '.bak')
        with open(path, 'w') as f:
            f.write(text)
        say('wrote %s (previous version kept as arena.yaml.bak)' % path)
    else:
        say(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
