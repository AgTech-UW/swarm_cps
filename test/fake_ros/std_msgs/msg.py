from builtin_interfaces.msg import Time
from msgbase import Msg


class String(Msg):
    FIELDS = (('data', str),)


class Header(Msg):
    FIELDS = (('stamp', Time), ('frame_id', str))
