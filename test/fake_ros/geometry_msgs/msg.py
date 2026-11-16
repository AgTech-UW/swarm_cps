from msgbase import Msg
from std_msgs.msg import Header


class Vector3(Msg):
    FIELDS = (('x', float), ('y', float), ('z', float))


class Point(Msg):
    FIELDS = (('x', float), ('y', float), ('z', float))


class Quaternion(Msg):
    FIELDS = (('x', float), ('y', float), ('z', float), ('w', float))

    def __init__(self, **kw):
        super().__init__(**kw)
        if 'w' not in kw:
            object.__setattr__(self, 'w', 1.0)


class Pose(Msg):
    FIELDS = (('position', Point), ('orientation', Quaternion))


class Twist(Msg):
    FIELDS = (('linear', Vector3), ('angular', Vector3))


class PoseStamped(Msg):
    FIELDS = (('header', Header), ('pose', Pose))


class TwistStamped(Msg):
    FIELDS = (('header', Header), ('twist', Twist))
