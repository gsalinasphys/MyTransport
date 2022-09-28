import os
import sys

import numpy as np
import sympy
from matplotlib import pyplot as plt
from sympy import atan2, atanh, cos, sin, sqrt, symbols, tanh
from sympy.diffgeom import (BaseCovarDerivativeOp, CoordSystem,
                            CovarDerivativeOp, Manifold, Patch)
from sympy.diffgeom import TensorProduct as TP
from sympy.diffgeom import metric_to_Christoffel_2nd
from sympy.utilities import lambdify

location = "/home/gsalinas/GitHub/MyTransport/PyTransport" # This should be the location of the PyTransport folder folder
sys.path.append(location) # Sets up python path to give access to PyTransSetup

import PyTransSetup

PyTransSetup.pathSet()  # This adds the other paths that PyTransport uses to the python path

import PyTransAngular as PyT
import PyTransScripts as PyS
