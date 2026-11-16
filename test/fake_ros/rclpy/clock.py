class _Clock:
    def __init__(self):
        self.t = 1000.0


CLOCK = _Clock()
EPOCH = 1.7e9


class FakeTimeModule:
    """Replaces the `time` module inside node modules so tests run in fake time."""
    @staticmethod
    def monotonic():
        return CLOCK.t

    @staticmethod
    def time():
        return EPOCH + CLOCK.t

    @staticmethod
    def sleep(s):
        CLOCK.t += s

    @staticmethod
    def strftime(fmt, *a):
        import time as _t
        return _t.strftime(fmt, *a)
