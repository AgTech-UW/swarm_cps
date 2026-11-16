"""Camera capture straight from V4L2, in its own thread, always handing out the
newest frame. No ROS and no image ever goes over DDS.

Why not usb_cam -> image topic -> apriltag_ros: at 1920x1080 that pipeline
decodes every MJPEG frame to 6 MB of RGB, re-encodes it to JPEG for the
compressed transport, sends it through DDS and decodes it again in the
detector. Each of those steps can drop or delay a frame, the double JPEG
softens the tag corners, and if DDS is bound to the Wi-Fi interface the image
traffic competes with the rovers' commands. Here the frame goes from the
driver to the detector inside one process, and only poses are published.
"""
import subprocess
import threading
import time

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None


def find_device(hint='See3CAM'):
    """First /dev/video node of the camera whose name contains `hint`, or None."""
    try:
        out = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    take = False
    for line in out.splitlines():
        if line and not line[0].isspace():
            take = hint.lower() in line.lower()
        elif take and '/dev/video' in line:
            return line.strip()
    return None


def set_controls(device, controls, log=print):
    """Apply V4L2 controls with v4l2-ctl and read each one back. Returns a
    dict name -> (wanted, actual or None)."""
    res = {}
    for name, want in (controls or {}).items():
        try:
            subprocess.run(['v4l2-ctl', '-d', str(device), '--set-ctrl=%s=%s' % (name, want)],
                           capture_output=True, text=True, timeout=5)
            got = subprocess.run(['v4l2-ctl', '-d', str(device), '--get-ctrl=%s' % name],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            val = got.split(':')[-1].strip() if got else None
            res[name] = (want, val)
            if val is None or str(val) != str(want):
                log('camera control %s: asked for %s, camera reports %s' % (name, want, val))
        except Exception as e:
            res[name] = (want, None)
            log('camera control %s could not be set (%s)' % (name, e))
        time.sleep(0.15)
    return res


class Capture:
    def __init__(self, device='auto', width=1920, height=1080, fps=30, fourcc='MJPG', controls=None,
                 control_delay=2.5, log=print):
        self.device = device
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.fourcc = fourcc
        self.controls = dict(controls or {})
        self.control_delay = control_delay
        self.log = log
        self.cap = None
        self._lock = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._run = False
        self._thread = None
        self.actual = {}
        self.error = None
        # health
        self.n_frames = 0
        self.n_late = 0             # intervals longer than 1.6 frame periods: the camera or USB dropped a frame
        self.n_fail = 0
        self.max_gap = 0.0
        self._t_prev = None
        self._t0 = None

    def open(self):
        if cv2 is None:
            raise RuntimeError('OpenCV is not available')
        dev = self.device
        if dev in (None, '', 'auto'):
            dev = find_device() or '/dev/video0'
        self.device = dev
        try:
            src = int(dev)
        except (TypeError, ValueError):
            src = dev
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError('cannot open camera %s. Is it plugged in, passed into the container '
                               '(--device), and not held by another program (usb_cam, cheese)?' % dev)
        four = cv2.VideoWriter_fourcc(*self.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, four)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FOURCC, four)                    # some drivers fall back when the size changes
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        f = int(cap.get(cv2.CAP_PROP_FOURCC))
        self.actual = dict(device=str(dev), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=float(cap.get(cv2.CAP_PROP_FPS)),
                           fourcc=''.join(chr((f >> (8 * i)) & 0xFF) for i in range(4)))
        self.cap = cap
        if (self.actual['width'], self.actual['height']) != (self.width, self.height):
            raise RuntimeError('camera gave %dx%d instead of %dx%d (format %s). The lens calibration only holds at '
                               'the resolution it was made at.' % (self.actual['width'], self.actual['height'],
                                                                   self.width, self.height, self.actual['fourcc']))
        return self.actual

    def start(self):
        if self.cap is None:
            self.open()
        self._run = True
        self._thread = threading.Thread(target=self._loop, name='camera', daemon=True)
        self._thread.start()

    def _loop(self):
        applied = not self.controls
        t_open = time.monotonic()
        fails = 0
        while self._run:
            ok = self.cap.grab()
            t = time.monotonic()                              # as close to arrival as we can stamp
            if ok:
                ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                fails += 1
                self.n_fail += 1
                if fails > 30:
                    self.error = 'camera stopped delivering frames'
                    self.log(self.error + '; reopening')
                    try:
                        self.cap.release()
                        time.sleep(1.0)
                        self.cap = None
                        self.open()
                        applied = not self.controls
                        t_open = time.monotonic()
                        self.error = None
                    except Exception as e:
                        self.error = str(e)
                        time.sleep(1.0)
                    fails = 0
                else:
                    time.sleep(0.005)
                continue
            fails = 0
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._t_prev is not None:
                gap = t - self._t_prev
                if self.n_frames > 30:                        # ignore start-up
                    self.max_gap = max(self.max_gap, gap)
                    if gap > 1.6 / self.fps:
                        self.n_late += 1
            else:
                self._t0 = t
            self._t_prev = t
            self.n_frames += 1
            with self._lock:
                self._frame, self._t = frame, t
                self._seq += 1
                self._lock.notify_all()
            if not applied and t - t_open > self.control_delay:
                # exposure and gain are set once the stream is running; set earlier, the
                # driver resets them (the lesson of the lab's camera guide)
                set_controls(self.device, self.controls, self.log)
                applied = True

    def wait_new(self, last_seq, timeout=0.5):
        """Newest frame with a sequence number above last_seq, or None on
        timeout. Returns (seq, t_monotonic, gray). Frames nobody asked for in
        time are skipped, never queued."""
        with self._lock:
            if self._seq <= last_seq:
                self._lock.wait(timeout)
            if self._seq <= last_seq or self._frame is None:
                return None
            return self._seq, self._t, self._frame

    def poll(self, last_seq):
        with self._lock:
            if self._seq <= last_seq or self._frame is None:
                return None
            return self._seq, self._t, self._frame

    def measured_fps(self):
        if self._t0 is None or self._t_prev is None or self._t_prev <= self._t0:
            return 0.0
        return (self.n_frames - 1) / (self._t_prev - self._t0)

    def stop(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
