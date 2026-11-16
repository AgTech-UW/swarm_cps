"""Typed message objects that behave like generated ROS 2 Python messages:
fixed fields, and each setter checks the type the way rclpy does."""
import copy


class Msg:
    FIELDS = ()          # (name, type or factory)

    def __init__(self, **kw):
        for name, typ in self.FIELDS:
            object.__setattr__(self, name, _default(typ))
        for k, v in kw.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        for fname, typ in self.FIELDS:
            if fname == name:
                _typecheck(type(self).__name__, name, typ, value)
                object.__setattr__(self, name, value)
                return
        raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))

    def _check(self):
        for name, typ in self.FIELDS:
            v = getattr(self, name)
            if isinstance(v, Msg):
                v._check()

    def __deepcopy__(self, memo):
        new = type(self)()
        for name, _ in self.FIELDS:
            object.__setattr__(new, name, copy.deepcopy(getattr(self, name), memo))
        return new


def _default(typ):
    if typ is float:
        return 0.0
    if typ is int or typ == 'uint32':
        return 0
    if typ is str:
        return ''
    if typ is bytes:
        return b''
    if typ is list:
        return []
    return typ()


def _typecheck(cls, name, typ, value):
    if typ is float:
        if not isinstance(value, float):
            raise AssertionError("The '%s' field of %s must be of type 'float', got %s" % (name, cls, type(value).__name__))
    elif typ is int:
        if not isinstance(value, int) or isinstance(value, bool) or not (-2 ** 31 <= value < 2 ** 31):
            raise AssertionError("The '%s' field of %s must be an int32, got %r (%s)" % (name, cls, value, type(value).__name__))
    elif typ == 'uint32':
        if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value < 2 ** 32):
            raise AssertionError("The '%s' field of %s must be a uint32, got %r" % (name, cls, value))
    elif typ is str:
        if not isinstance(value, str):
            raise AssertionError("The '%s' field of %s must be of type 'str'" % (name, cls))
    elif typ in (bytes, list):
        pass
    elif not isinstance(value, typ):
        raise AssertionError("The '%s' field of %s must be a %s" % (name, cls, typ.__name__))
