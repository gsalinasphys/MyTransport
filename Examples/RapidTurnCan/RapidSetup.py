####################################### Setup file for the rapid turn model ######################################################

import sys
from math import atan, sqrt

import sympy as sym
from gravipy.tensorial import *

location = "/home/gsalinas/GitHub/PyTransport/PyTransport" # this should be the location of the PyTransport folder
sys.path.append(location)  # we add this location to the python path

import PyTransSetup  # the above commands allows python to find the PyTransSetup module and import it

nF = 2  # number of fields
nP = 4  # number of parameters
f = sym.symarray('f',nF)   # an array representing the nF fields present for this model
p = sym.symarray('p',nP)   # an array representing the nP parameters needed to define this model, format [V0, alpha, m, rho0]

V = p[0] - p[1]*sym.atan(f[1]/f[0]) + 0.5*p[2]**2*(sym.sqrt(f[0]**2 + f[1]**2) - p[3])**2   # this is the potential written in sympy notation

PyTransSetup.potential(V, nF, nP) # writes this potential and its derivatives into C++ file potential.h when run

PyTransSetup.compileName3("RapidCan", True) # this compiles a python module using the C++ code, including the edited potential.h file, called PyTransRapid
                                 # and places it in the location folder, ready for use

############################################################################################################################################

