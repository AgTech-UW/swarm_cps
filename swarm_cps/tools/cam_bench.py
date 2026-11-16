"""Is the camera link good enough? Run this before anything else, with no ROS
node running. It opens the camera exactly as the perception node will, counts
late and missing frames for a while, times the tag detector on real frames,
and says PASS or FAIL for each thing that has caused dropped frames before.

    cam_bench                    # the mode in camera.yaml, 10 s
    cam_bench --modes            # try 1080p at 30, 60 and 120 fps and report what the link sustains
    cam_bench --seconds 60       # soak test
"""
import argparse
import subprocess
import sys
import time

import numpy as np

from .common import cv2, make_detector, open_camera, say, settings, verdict


def usb_speed(device):
    """Negotiated USB speed of a /dev/video device in Mbit/s, or None."""
    try:
        name = str(device).split('/')[-1]
        out = subprocess.run(['udevadm', 'info', '-q', 'path', '-n', '/dev/' + name], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        parts = ('/sys' + out).split('/')
        for k in range(len(parts), 3, -1):
            try:
                with open('/'.join(parts[:k]) + '/speed') as f:
                    return int(float(f.read().strip()))
            except Exception:
                continue
    except Exception:
        pass
    return None


def run_mode(st, fps, seconds, device=None, detect=True):
    cap = open_camera(st, device, fps=fps, warmup=3.0)
    det = make_detector(st) if detect else None
    n0, late0, t0 = cap.n_frames, cap.n_late, time.monotonic()
    cap.max_gap = 0.0
    seq, times, ids, cells, gray = 0, [], {}, {}, None
    try:
        while time.monotonic() - t0 < seconds:
            got = cap.wait_new(seq, 1.0)
            if got is None:
                continue
            seq, _, gray = got
            if det is not None:
                a = time.monotonic()
                ds = det.detect(gray)
                times.append(time.monotonic() - a)
                for d in ds:
                    ids[d.id] = ids.get(d.id, 0) + 1
                    cells.setdefault(d.id, []).append(d.min_side_px() / 8.0)
        el = time.monotonic() - t0
        res = dict(fps=(cap.n_frames - n0) / el, late=cap.n_late - late0, fails=cap.n_fail, max_gap=cap.max_gap,
                   processed=len(times), detect_ms=1e3 * float(np.mean(times)) if times else None,
                   detect_p95_ms=1e3 * float(np.percentile(times, 95)) if times else None, ids=ids,
                   cells={i: float(np.mean(v)) for i, v in cells.items()}, device=cap.device, actual=cap.actual,
                   backend=det.backend if det else None, frame=gray, seconds=el)
    finally:
        cap.stop()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog='cam_bench', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config-dir', dest='config_dir', default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--fps', type=float, default=None)
    ap.add_argument('--modes', action='store_true')
    ap.add_argument('--snapshot', default='/tmp/swarm_bench.jpg')
    args = ap.parse_args(argv)
    st = settings(args)
    cc = st.camera.get('capture') or {}
    if args.modes:
        say('Trying 1080p at 30, 60 and 120 fps, 5 s each (capture only).')
        for fps in (30, 60, 120):
            r = run_mode(st, fps, 5.0, args.device, detect=False)
            say('  asked %3d fps -> got %5.1f fps, %d late frames, longest gap %.0f ms'
                % (fps, r['fps'], r['late'], 1e3 * r['max_gap']))
        say('Set capture.fps in camera.yaml to the highest rate that holds with no late frames. The detector '
            'also has to keep up; run cam_bench without --modes at that rate.')
        return 0
    fps = args.fps or float(cc.get('fps', 30))
    say('Measuring for %.0f s at %.0f fps ...' % (args.seconds, fps))
    r = run_mode(st, fps, args.seconds, args.device)
    ok = True
    say('')
    sp = usb_speed(r['device'])
    if sp is not None:
        ok &= verdict(sp >= 5000, 'USB link speed %d Mbit/s%s' % (sp, '' if sp >= 5000 else
                      ': the camera is on a USB 2 port, hub or cable. Use a blue USB 3 port and a short or ACTIVE USB 3 cable.'))
    else:
        say('  [ -- ] USB link speed unknown (inside a container?). On the host: lsusb -t  should show 5000M for the camera.')
    ok &= verdict(r['actual'].get('fourcc') == cc.get('fourcc', 'MJPG'), 'pixel format %s' % r['actual'].get('fourcc'))
    ok &= verdict(r['fps'] >= 0.93 * fps, 'camera delivers %.1f fps (asked %.0f)%s' % (r['fps'], fps, '' if r['fps'] >= 0.93 * fps else
                  ': exposure longer than a frame, USB 2 link, or CPU too busy to decode'))
    n = max(1, int(r['fps'] * r['seconds']))
    ok &= verdict(r['late'] <= max(1, 0.005 * n), '%d late or missing frames out of %d; longest gap %.0f ms'
                  % (r['late'], n, 1e3 * r['max_gap']))
    ok &= verdict(r['fails'] == 0, '%d failed reads' % r['fails'])
    if r['detect_ms'] is not None:
        say('  detector: %s, full frame %.0f ms average, %.0f ms 95th percentile' % (r['backend'], r['detect_ms'], r['detect_p95_ms']))
        say('            (a full search runs every 5 s, and 5 times a second while a rover is lost; tracked rovers cost a few ms each)')
        verdict(r['backend'] == 'apriltag', 'AprilTag library in use' if r['backend'] == 'apriltag' else
                'using the OpenCV fallback detector: pip install --no-deps pupil-apriltags')
        ok &= verdict(r['detect_ms'] < 120, 'full-frame search under 120 ms')
        if r['ids']:
            say('  tags seen (id: %% of frames, pixels per tag cell on the short side):')
            for i in sorted(r['ids']):
                c = r['cells'][i]
                say('    %4d: %3.0f %%   %.1f px/cell %s' % (i, 100.0 * r['ids'][i] / max(1, r['processed']), c,
                                                           '' if c >= 2.5 else ('(marginal)' if c >= 1.8 else '(too small)')))
        else:
            say('  no tags in view')
    if r['frame'] is not None:
        mean = float(r['frame'].mean())
        sat = float((r['frame'] > 250).mean())
        verdict(40 < mean < 200 and sat < 0.05, 'image brightness %.0f/255, %.1f %% saturated%s' % (mean, 100 * sat,
                '' if 40 < mean < 200 and sat < 0.05 else ': adjust exposure_time_absolute / gain in camera.yaml'))
        cv2.imwrite(args.snapshot, r['frame'])
        say('  snapshot: %s' % args.snapshot)
    say('')
    say('RESULT: %s' % ('camera link is good' if ok else 'fix the FAIL lines before going further'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
