from msgbase import Msg


class Time(Msg):
    FIELDS = (('sec', int), ('nanosec', 'uint32'))
