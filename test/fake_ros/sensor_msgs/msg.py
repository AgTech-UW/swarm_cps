from msgbase import Msg
from std_msgs.msg import Header


class CompressedImage(Msg):
    FIELDS = (('header', Header), ('format', str), ('data', bytes))
