import importlib.metadata

__version__ = importlib.metadata.version("lqg")

from lqg.ccg import xcorr
from lqg.system import LQG, Actor, Dynamics, System
from lqg.spec import LQGSpec
