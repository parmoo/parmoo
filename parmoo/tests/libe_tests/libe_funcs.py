# Define a 5d problem with 3 objectives
n = 5
o = 3

def dtlz2_sim_unnamed(x):
    """ Evaluates the sim function for a collection of points given in
    ``H['x']``.

    """

    import math
    import numpy as np


    # Create output array for sim outs
    f = np.zeros(o)
    # Compute the kernel function g(x)
    gx = np.dot(x[o-1:n]-0.5, x[o-1:n]-0.5)
    # Compute the simulation outputs
    f[0] = (1.0 + gx)
    for y in x[:o-1]:
        f[0] *= math.cos(math.pi * y / 2.0)
    for i in range(1, o):
        f[i] = (1.0 + gx) * math.sin(math.pi * x[o-1-i] / 2.0)
        for y in x[:o-1-i]:
            f[i] *= math.cos(math.pi * y / 2.0)
    return f
    
def obj1_unnamed(x, s): return s[0]
def obj2_unnamed(x, s): return s[1]
def obj3_unnamed(x, s): return s[2]

# Define functions for named runs

def dtlz2_sim_named(x):
    """ Evaluates the sim function for a collection of points given in
    ``H['x']``.

    """

    import numpy as np

    # Unpack names into array
    xx = np.zeros(n)
    names = [f"x{i+1}" for i in range(n)]
    for i, name in enumerate(names):
        xx[i] = x[name]
    # Use dtlz2_sim to evaluate
    return dtlz2_sim_unnamed(xx)

def obj1_named(x, s): return s['DTLZ2'][0]
def obj2_named(x, s): return s['DTLZ2'][1]
def obj3_named(x, s): return s['DTLZ2'][2]
