import copy

from . import node as _node
from . import qos as _qos   # noqa: F401
from .clock import CLOCK    # noqa: F401


class Context:
    def __init__(self):
        self.domain_id = None
        self._ok = False

    def ok(self):
        return self._ok


_default = Context()
BUS = {}          # domain -> topic -> dict(type, subs=[...], pubs=[...], last=msg)
NODES = []


def reset():
    global _default
    BUS.clear()
    del NODES[:]
    _default = Context()
    CLOCK.t = 1000.0


def get_default_context():
    return _default


def init(*, args=None, context=None, domain_id=None, signal_handler_options=None):
    import os
    ctx = context if context is not None else _default
    if ctx._ok:
        raise RuntimeError('Context.init() must only be called once')
    ctx.domain_id = int(domain_id) if domain_id is not None else int(os.environ.get('ROS_DOMAIN_ID', '0') or 0)
    ctx._ok = True


def ok(*, context=None):
    return (context if context is not None else _default)._ok


def shutdown(*, context=None):
    ctx = context if context is not None else _default
    if not ctx._ok:
        raise RuntimeError('Context must be initialized before it can be shutdown')
    ctx._ok = False


def create_node(node_name, *, context=None, cli_args=None, namespace=None, use_global_arguments=True,
                enable_rosout=True, start_parameter_services=True, parameter_overrides=None,
                allow_undeclared_parameters=False, automatically_declare_parameters_from_overrides=False):
    return _node.Node(node_name, context=context, namespace=namespace, parameter_overrides=parameter_overrides)


def spin(node, executor=None):
    return                                       # tests drive the timers themselves


def spin_once(node, *, executor=None, timeout_sec=None):
    run_for(timeout_sec or 0.0, nodes=[node])


def run_for(seconds, step=None, nodes=None):
    """Advance the fake clock, firing every due timer in time order."""
    end = CLOCK.t + seconds
    while True:
        timers = [t for n in (nodes or NODES) if not n._destroyed for t in n._timers]
        due = [t for t in timers if t.next <= end + 1e-12]
        if not due:
            break
        t = min(due, key=lambda t: t.next)
        CLOCK.t = max(CLOCK.t, t.next)
        t.next += t.period
        t.callback()
    CLOCK.t = max(CLOCK.t, end)


def deliver(domain, topic, msg, pub):
    ent = BUS.setdefault(domain, {}).setdefault(topic, dict(type=type(msg), subs=[], pubs=[], last={}))
    for sub in list(ent['subs']):
        if sub.msg_type is not type(msg):
            raise TypeError('topic %s: publisher sends %s, subscriber expects %s' % (topic, type(msg).__name__, sub.msg_type.__name__))
        if _qos.compatible(pub.qos, sub.qos):
            sub.callback(copy.deepcopy(msg))
