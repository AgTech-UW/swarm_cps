import enum


class ReliabilityPolicy(enum.IntEnum):
    SYSTEM_DEFAULT = 0
    RELIABLE = 1
    BEST_EFFORT = 2


class DurabilityPolicy(enum.IntEnum):
    SYSTEM_DEFAULT = 0
    TRANSIENT_LOCAL = 1
    VOLATILE = 2


class HistoryPolicy(enum.IntEnum):
    SYSTEM_DEFAULT = 0
    KEEP_LAST = 1
    KEEP_ALL = 2


class QoSProfile:
    def __init__(self, *, history=None, depth=None, reliability=ReliabilityPolicy.RELIABLE,
                 durability=DurabilityPolicy.VOLATILE, lifespan=None, deadline=None, liveliness=None,
                 liveliness_lease_duration=None, avoid_ros_namespace_conventions=False):
        if history is None and depth is None:
            raise TypeError('QoSProfile needs history and/or depth settings')
        for v, t in ((reliability, ReliabilityPolicy), (durability, DurabilityPolicy)):
            if not isinstance(v, t):
                raise TypeError('bad QoS policy %r' % (v,))
        self.history, self.depth, self.reliability, self.durability = history, depth, reliability, durability


qos_profile_sensor_data = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
qos_profile_system_default = QoSProfile(depth=10)


def to_profile(q):
    if isinstance(q, int):
        return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=q)
    if isinstance(q, QoSProfile):
        return q
    raise TypeError('qos_profile must be an int or a QoSProfile')


def compatible(pub, sub):
    """The DDS request-versus-offered rules."""
    if sub.reliability == ReliabilityPolicy.RELIABLE and pub.reliability == ReliabilityPolicy.BEST_EFFORT:
        return False
    if sub.durability == DurabilityPolicy.TRANSIENT_LOCAL and pub.durability == DurabilityPolicy.VOLATILE:
        return False
    return True
