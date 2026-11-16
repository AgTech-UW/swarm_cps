"""Shared by the command-line tools: getting frames and detections."""
import sys
import time

from ..capture import Capture
from ..core.tags import Detector
from ..settings import Settings

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


def say(msg=''):
    print(msg, flush=True)


def verdict(ok, text):
    say('  [%s] %s' % ('PASS' if ok else 'FAIL', text))
    return bool(ok)


def make_detector(st, decimate=None):
    dc = st.camera.get('detector') or {}
    return Detector(dc.get('backend', 'auto'), decimate=float(decimate if decimate is not None else 1.0),
                    threads=int(dc.get('threads', 4)), min_margin=float(dc.get('min_margin', 25)))


def open_camera(st, device=None, fps=None, size=None, warmup=3.5):
    cc = st.camera.get('capture') or {}
    w, h = size or (cc.get('width', 1920), cc.get('height', 1080))
    cap = Capture(device or cc.get('device', 'auto'), w, h, fps or cc.get('fps', 30), cc.get('fourcc', 'MJPG'),
                  cc.get('controls'), log=lambda s: say('  ! ' + str(s)))
    try:
        info = cap.open()
    except Exception as e:
        say('CAMERA: %s' % e)
        say('If "ros2 run swarm_cps perception" or usb_cam is running, stop it first: one program at a time can '
            'hold the camera.')
        sys.exit(2)
    say('camera %(device)s: %(width)dx%(height)d %(fourcc)s, driver says %(fps).0f fps' % info)
    cap.start()
    t0 = time.monotonic()
    seq = 0
    while time.monotonic() - t0 < warmup:                     # let exposure settings take effect
        got = cap.wait_new(seq, 1.0)
        if got:
            seq = got[0]
    return cap


def grab_detections(st, n=30, image=None, device=None, detector=None):
    """n frames of detections, from a live camera or one image file.
    Returns (list of dict id -> TagDetection, last gray frame)."""
    det = detector or make_detector(st)
    if image:
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            say('cannot read image %s' % image)
            sys.exit(2)
        return [{d.id: d for d in det.detect(gray)}], gray
    cap = open_camera(st, device)
    frames, seq, gray = [], 0, None
    try:
        while len(frames) < n:
            got = cap.wait_new(seq, 2.0)
            if got is None:
                say('camera stopped delivering frames')
                break
            seq, _, gray = got
            frames.append({d.id: d for d in det.detect(gray)})
    finally:
        cap.stop()
    return frames, gray


def settings(args):
    try:
        return Settings(getattr(args, 'config_dir', None))
    except Exception as e:
        say('CONFIG: %s' % e)
        sys.exit(2)
