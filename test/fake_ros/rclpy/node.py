import copy

from . import qos as _qos
from .clock import CLOCK, EPOCH


class _Time:
    def __init__(self, ns):
        self.nanoseconds = int(ns)

    def to_msg(self):
        from builtin_interfaces.msg import Time
        return Time(sec=self.nanoseconds // 10 ** 9, nanosec=self.nanoseconds % 10 ** 9)


class _ClockObj:
    def now(self):
        return _Time((EPOCH + CLOCK.t) * 1e9)


class _Logger:
    def __init__(self, name):
        self.name = name
        self.records = []

    def _log(self, level, msg, throttle_duration_sec=None, skip_first=False, once=False):
        if not isinstance(msg, str):
            raise TypeError('log message must be a str, got %s' % type(msg).__name__)
        self.records.append((level, msg))

    def debug(self, msg, **kw):
        self._log('debug', msg, **kw)

    def info(self, msg, **kw):
        self._log('info', msg, **kw)

    def warn(self, msg, **kw):
        self._log('warn', msg, **kw)

    warning = warn

    def error(self, msg, **kw):
        self._log('error', msg, **kw)

    def fatal(self, msg, **kw):
        self._log('fatal', msg, **kw)


class _Param:
    def __init__(self, name, value):
        self.name, self.value = name, value


class _Timer:
    def __init__(self, period, callback):
        self.period, self.callback = float(period), callback
        self.next = CLOCK.t + self.period

    def cancel(self):
        self.next = float('inf')


class Publisher:
    def __init__(self, node, msg_type, topic, qos):
        self.node, self.msg_type, self.topic, self.qos = node, msg_type, topic, qos
        self.sent = []

    def publish(self, msg):
        if type(msg) is not self.msg_type:
            raise TypeError('publishing %s on %s, which is a %s topic' % (type(msg).__name__, self.topic, self.msg_type.__name__))
        import rclpy
        msg._check()
        self.sent.append(copy.deepcopy(msg))
        ent = rclpy.BUS.setdefault(self.node._domain, {}).setdefault(self.topic, dict(type=self.msg_type, subs=[], pubs=[], last={}))
        ent['last'][id(self)] = (self, copy.deepcopy(msg))
        rclpy.deliver(self.node._domain, self.topic, msg, self)

    def get_subscription_count(self):
        import rclpy
        ent = rclpy.BUS.get(self.node._domain, {}).get(self.topic)
        return len([s for s in ent['subs'] if _qos.compatible(self.qos, s.qos)]) if ent else 0


class Subscription:
    def __init__(self, msg_type, topic, callback, qos):
        self.msg_type, self.topic, self.callback, self.qos = msg_type, topic, callback, qos


class Node:
    def __init__(self, node_name, *, context=None, cli_args=None, namespace=None, use_global_arguments=True,
                 enable_rosout=True, start_parameter_services=True, parameter_overrides=None,
                 allow_undeclared_parameters=False, automatically_declare_parameters_from_overrides=False):
        import rclpy
        ctx = context if context is not None else rclpy.get_default_context()
        if not ctx.ok():
            raise RuntimeError('rclpy.init() has not been called for this context')
        self._name = node_name
        self._domain = ctx.domain_id
        self._params = {}
        self._overrides = dict(getattr(Node, 'OVERRIDES', {}).get(node_name, {}))
        self._timers = []
        self._pubs = []
        self._subs = []
        self._logger = _Logger(node_name)
        self._destroyed = False
        rclpy.NODES.append(self)

    def get_name(self):
        return self._name

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return _ClockObj()

    def declare_parameter(self, name, value=None, descriptor=None, ignore_override=False):
        if name in self._params:
            raise RuntimeError('parameter %r already declared' % name)
        if isinstance(value, (list, tuple)) and len(value) == 0:
            raise TypeError('empty list default has no type in Humble')
        v = self._overrides.get(name, value)
        if value is not None and v is not None and type(v) is not type(value):
            raise TypeError('parameter %r: override %r is not a %s' % (name, v, type(value).__name__))
        self._params[name] = _Param(name, v)
        return self._params[name]

    def get_parameter(self, name):
        if name not in self._params:
            raise KeyError('ParameterNotDeclaredException: %s' % name)
        return self._params[name]

    def create_publisher(self, msg_type, topic, qos_profile, *, callback_group=None, event_callbacks=None,
                         qos_overriding_options=None, publisher_class=None):
        if not topic.startswith('/'):
            raise ValueError('tests expect absolute topic names, got %r' % topic)
        p = Publisher(self, msg_type, topic, _qos.to_profile(qos_profile))
        import rclpy
        ent = rclpy.BUS.setdefault(self._domain, {}).setdefault(topic, dict(type=msg_type, subs=[], pubs=[], last={}))
        if ent['type'] is not msg_type:
            raise TypeError('topic %s already has type %s' % (topic, ent['type'].__name__))
        ent['pubs'].append(p)
        self._pubs.append(p)
        return p

    def create_subscription(self, msg_type, topic, callback, qos_profile, *, callback_group=None, event_callbacks=None,
                            qos_overriding_options=None, raw=False):
        if not callable(callback):
            raise TypeError('callback must be callable')
        s = Subscription(msg_type, topic, callback, _qos.to_profile(qos_profile))
        import rclpy
        ent = rclpy.BUS.setdefault(self._domain, {}).setdefault(topic, dict(type=msg_type, subs=[], pubs=[], last={}))
        if ent['type'] is not msg_type:
            raise TypeError('topic %s already has type %s' % (topic, ent['type'].__name__))
        ent['subs'].append(s)
        self._subs.append(s)
        if s.qos.durability == _qos.DurabilityPolicy.TRANSIENT_LOCAL:       # late joiner gets the latched sample
            for pub, msg in list(ent['last'].values()):
                if pub.qos.durability == _qos.DurabilityPolicy.TRANSIENT_LOCAL and _qos.compatible(pub.qos, s.qos):
                    callback(copy.deepcopy(msg))
        return s

    def create_timer(self, timer_period_sec, callback, callback_group=None, clock=None):
        t = _Timer(timer_period_sec, callback)
        self._timers.append(t)
        return t

    def count_subscribers(self, topic):
        import rclpy
        ent = rclpy.BUS.get(self._domain, {}).get(topic)
        return len(ent['subs']) if ent else 0

    def count_publishers(self, topic):
        import rclpy
        ent = rclpy.BUS.get(self._domain, {}).get(topic)
        return len(ent['pubs']) if ent else 0

    def destroy_node(self):
        import rclpy
        self._destroyed = True
        for ent in rclpy.BUS.get(self._domain, {}).values():
            ent['subs'] = [s for s in ent['subs'] if s not in self._subs]
            ent['pubs'] = [p for p in ent['pubs'] if p not in self._pubs]
        self._timers = []
