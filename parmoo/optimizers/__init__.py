from .pattern_search import GlobalSurrogate_PS, LocalSurrogate_PS
from .random_search import GlobalSurrogate_RS
from .lbfgsb import GlobalSurrogate_BFGS, LocalSurrogate_BFGS

LocalGPS = LocalSurrogate_PS
GlobalGPS = GlobalSurrogate_PS
RandomSearch = GlobalSurrogate_RS
TR_BFGSB = LocalSurrogate_BFGS
BFGSB = GlobalSurrogate_BFGS
