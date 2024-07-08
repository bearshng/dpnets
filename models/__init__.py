from .DPNetU import Params
from .DPNetU import DPNetU
from .DPNetS import DPNetS

"""Define commonly used architecture"""


def mscnet(params):
    net = DPNetU(params)
    net.use_2dconv = False
    net.bandwise = False
    return net
def mscnet_l1(params):
    net = DPNetS(params)
    net.use_2dconv = False
    net.bandwise = False
    return net
