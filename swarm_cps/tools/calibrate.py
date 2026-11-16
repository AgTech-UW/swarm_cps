"""Floor calibration and its checks.

  calibrate floor      solve where the camera is from the floor reference tags
                       (camera.yaml "anchors") and write camera_pose.yaml
  calibrate floor --walk
                       no reference tags printed? carry ONE tag to each marked
                       spot instead; you are prompted spot by spot
  calibrate check      print where every visible tag is, with its measured size
  calibrate yaw        rovers parked all facing the same way: work out how each
                       tag is rotated on its rover (fleet.yaml yaw_offset_deg)
  calibrate snapshot   save one camera frame

Stop the perception node first; only one program can hold the camera.
Add --image FILE to work from a saved frame instead of the live camera.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

from ..core.calib import average_detections, calibrate_floor
from ..core.localize import wrap
from ..core.tags import tag_floor_pose
from .common import cv2, grab_detections, say, settings, verdict


def _annotate(gray, st, fm, mean, path):
    from ..core.viz import draw_floor_grid
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ar = st.arena()
    draw_floor_grid(img, fm, (ar.x0, ar.x1, ar.y0, ar.y1))
    for i, (c, n, jit) in mean.items():
        p = c.mean(0)
        cv2.polylines(img, [c.astype(np.int32)], True, (0, 220, 0), 2)
        cv2.putText(img, str(i), (int(p[0]) + 10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path


def cmd_floor(args):
    st = settings(args)
    cam = st.camera_model()
    if args.walk:
        return cmd_walk(args, st, cam)
    anchors = st.anchors()
    if len(anchors) < 4:
        say('camera.yaml lists %d floor reference tags; at least 4 are needed, 6 or more recommended.' % len(anchors))
        return 2
    say('Reading %d frames. Keep the floor still and clear of people.' % args.frames)
    frames, gray = grab_detections(st, args.frames, args.image, args.device)
    mean = average_detections(frames)
    seen = sorted(mean)
    say('tags seen: %s' % (seen or 'none'))
    try:
        fm, rep = calibrate_floor(cam, mean, anchors)
    except Exception as e:
        say('CALIBRATION FAILED: %s' % e)
        if gray is not None:
            cv2.imwrite(args.out_image, gray)
            say('saved what the camera sees to %s; are the reference tags in view, flat, and unobstructed?' % args.out_image)
        return 1
    m = fm.describe()
    say('')
    say('camera position (%.2f, %.2f, %.2f) m; looking %.0f deg below horizontal, toward heading %.0f deg'
        % (m['position'][0], m['position'][1], m['position'][2], m['pitch_down_deg'], m['yaw_deg']))
    say('reference tags used: %s   missing: %s' % (rep['anchors_used'], rep['anchors_missing'] or 'none'))
    say('fit: %.2f px reprojection; %.1f mm on the floor (rms)' % (rep['reproj_rms_px'], 1e3 * rep['floor_rms_m']))
    if rep['leave_one_out_rms_m'] is not None:
        say('cross-check (each tag predicted from the others): %.1f mm rms, worst %.1f mm'
            % (1e3 * rep['leave_one_out_rms_m'], 1e3 * max(rep['leave_one_out_m'])))
        worst = int(np.argmax(rep['leave_one_out_m']))
        if max(rep['leave_one_out_m']) > 0.03:
            say('  worst tag: id %d. Re-measure its position first.' % rep['anchors_used'][worst])
    for p in rep['problems']:
        say('  ! ' + p)
    jit = max((j for _, _, j in mean.values()), default=0.0)
    if jit > 0.5:
        say('  ! tag corners jitter by %.2f px between frames: check focus, exposure and that the mount is rigid' % jit)
    ok = rep['grade'] != 'bad'
    verdict(ok, 'calibration grade: %s' % rep['grade'].upper())
    if gray is not None:
        say('picture with the 0.5 m floor grid drawn on it: %s' % _annotate(gray, st, fm, mean, args.out_image))
        say('  open it: the grid lines must lie on the floor and pass through your tape marks.')
    if ok or args.force:
        say('wrote %s' % st.save_floor_map(fm, rep))
        return 0
    say('NOT saved (grade bad). Fix the problems above, or use --force to save anyway.')
    return 1


def cmd_walk(args, st, cam):
    """One tag carried to each spot. Spots come from camera.yaml anchors (their
    ids are ignored); the tag may be at any constant height (--height)."""
    spots = list(st.anchors().values())
    if len(spots) < 4:
        say('camera.yaml needs at least 4 anchor positions to walk to')
        return 2
    say('Walk mode: put tag id %d flat at each spot in turn (height %.3f m).' % (args.tag, args.height))
    mean, anchors = {}, {}
    gray = None
    for k, a in enumerate(spots):
        try:
            input('  spot %d of %d: (%.2f, %.2f)  then press Enter (Ctrl+C to finish early) ' % (k + 1, len(spots), a['x'], a['y']))
        except (KeyboardInterrupt, EOFError):
            break
        frames, gray = grab_detections(st, 15, None, args.device)
        m = average_detections(frames)
        if args.tag not in m:
            say('    tag %d not seen here; skipped' % args.tag)
            continue
        mean[1000 + k] = m[args.tag]
        anchors[1000 + k] = dict(x=a['x'], y=a['y'], z=args.height)
        say('    ok')
    try:
        fm, rep = calibrate_floor(cam, mean, anchors)
    except Exception as e:
        say('CALIBRATION FAILED: %s' % e)
        return 1
    say('fit %.2f px; cross-check %s mm; grade %s' % (rep['reproj_rms_px'], '%.1f' % (1e3 * rep['leave_one_out_rms_m'])
                                                     if rep['leave_one_out_rms_m'] is not None else 'n/a', rep['grade'].upper()))
    for p in rep['problems']:
        say('  ! ' + p)
    if rep['grade'] != 'bad' or args.force:
        say('wrote %s' % st.save_floor_map(fm, rep))
        return 0
    return 1


def _need_map(st):
    fm = st.floor_map()
    if fm is None:
        say('No camera_pose.yaml yet. Run: calibrate floor')
        sys.exit(2)
    return fm


def cmd_check(args):
    st = settings(args)
    fm = _need_map(st)
    tags = {r.tag_id: r for r in st.robot_tags()}
    anchors = st.anchors()
    frames, _ = grab_detections(st, args.frames, args.image, args.device)
    acc = {}
    for f in frames:
        for i, d in f.items():
            if i in tags:
                h = tags[i].tag_height
            else:
                h = anchors.get(i, {}).get('z', 0.0)
            p = tag_floor_pose(d, fm, h, None)
            if p is not None:
                acc.setdefault(i, []).append((p['x'], p['y'], p['yaw'], p['size'], d.min_side_px() / 8.0))
    if not acc:
        say('no tags seen')
        return 1
    say('%-10s %-8s %8s %8s %8s %9s %9s %8s %s' % ('tag', 'what', 'x m', 'y m', 'yaw deg', 'size mm', 'jitter mm', 'px/cell', ''))
    ok = True
    for i in sorted(acc):
        a = np.array(acc[i])
        x, y = a[:, 0].mean(), a[:, 1].mean()
        yaw = math.atan2(np.sin(a[:, 2]).mean(), np.cos(a[:, 2]).mean())
        jit = 1e3 * float(np.hypot(a[:, 0].std(), a[:, 1].std()))
        note = ''
        what = 'floor'
        if i in tags:
            what = tags[i].name
            err = abs(a[:, 3].mean() - tags[i].tag_size) / tags[i].tag_size
            if err > 0.06:
                note = 'size off by %.0f %%: check tags.height / tags.size in fleet.yaml' % (100 * err)
                ok = False
        elif i in anchors:
            e = math.hypot(x - anchors[i]['x'], y - anchors[i]['y'])
            note = '%.0f mm from its surveyed position' % (1e3 * e)
            ok = ok and e < 0.04
        if a[:, 4].mean() < 1.8:
            note += ' | small in the image: detection will be unreliable here'
        say('%-10d %-8s %8.3f %8.3f %8.1f %9.1f %9.1f %8.1f %s' % (i, what, x, y, math.degrees(yaw), 1e3 * a[:, 3].mean(),
                                                                     jit, a[:, 4].mean(), note))
    say('')
    say('Put a rover on a tape mark you have measured and compare. Within 2 cm is good.')
    return 0 if ok else 1


def cmd_yaw(args):
    st = settings(args)
    fm = _need_map(st)
    tags = {r.tag_id: r for r in st.robot_tags()}
    frames, _ = grab_detections(st, args.frames, args.image, args.device)
    acc = {}
    for f in frames:
        for i, d in f.items():
            if i in tags:
                p = tag_floor_pose(d, fm, tags[i].tag_height, tags[i].tag_size)
                if p is not None:
                    acc.setdefault(i, []).append(p['yaw'])
    if not acc:
        say('no rover tags seen')
        return 1
    want = math.radians(args.heading)
    say('Assuming every rover is parked pointing along heading %.0f deg (0 = +x/east, 90 = +y/north).' % args.heading)
    path = os.path.join(st.dir, 'fleet.yaml')
    text = open(path).read()
    for i in sorted(acc):
        yaw = math.atan2(np.mean(np.sin(acc[i])), np.mean(np.cos(acc[i])))
        off = round(math.degrees(wrap(want - yaw)), 1) + 0.0
        name = tags[i].name
        say('  %-8s tag reads %7.1f deg -> yaw_offset_deg: %s' % (name, math.degrees(yaw), off))
        if args.write:
            pat = re.compile(r'^(\s*-\s*\{[^\n}]*name:\s*%s\b[^\n}]*)\}' % re.escape(name), re.M)
            m = pat.search(text)
            if m:
                line = m.group(1)
                line = re.sub(r'yaw_offset_deg:\s*[-+0-9.eE]+', 'yaw_offset_deg: %s' % off, line) if 'yaw_offset_deg' in line \
                    else line.rstrip() + ', yaw_offset_deg: %s' % off
                text = text[:m.start()] + line + '}' + text[m.end():]
            else:
                say('    could not find %s on one line in fleet.yaml; edit it by hand' % name)
    if args.write:
        open(path, 'w').write(text)
        say('updated %s' % path)
    else:
        say('Copy these into fleet.yaml, or run again with --write.')
    return 0


def cmd_snapshot(args):
    st = settings(args)
    frames, gray = grab_detections(st, 3, args.image, args.device)
    cv2.imwrite(args.out_image, gray)
    say('saved %s (%d tags visible: %s)' % (args.out_image, len(frames[-1]), sorted(frames[-1])))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='calibrate', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['floor', 'check', 'yaw', 'snapshot'])
    ap.add_argument('--config-dir', dest='config_dir', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--image', default=None, help='use this saved frame instead of the camera')
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--out-image', dest='out_image', default='/tmp/swarm_calibration.jpg')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--walk', action='store_true')
    ap.add_argument('--tag', type=int, default=1, help='walk mode: id of the tag you carry')
    ap.add_argument('--height', type=float, default=0.0, help='walk mode: height of that tag above the floor (m)')
    ap.add_argument('--heading', type=float, default=0.0, help='yaw: direction the rovers are parked facing (deg)')
    ap.add_argument('--write', action='store_true', help='yaw: write the offsets into fleet.yaml')
    args = ap.parse_args(argv)
    return dict(floor=cmd_floor, check=cmd_check, yaw=cmd_yaw, snapshot=cmd_snapshot)[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
