####################################### Setup file for the angular inflation model ######################################################

import sys

import sympy as sym
from gravipy.tensorial import *

location = "/home/gsalinas/GitHub/PyTransport/PyTransport" # this should be the location of the PyTransport folder
sys.path.append(location)  # we add this location to the python path

import PyTransSetup  # the above commands allows python to find the PyTransSetup module and import it

nF = 2  # number of fields
nP = 3  # number of parameters
f = sym.symarray('f',nF)   # an array representing the nF fields present for this model [phi, chi]
p = sym.symarray('p',nP)   # an array representing the nP parameters needed to define this model, format [alpha, R, mphi]

V = p[0]/2 * p[2]**2 * (f[0]**2 + p[1]*f[1]**2)   # this is the potential written in sympy notation
G = 6*p[0]/(1-f[0]**2-f[1]**2)**2 * sym.Matrix([[1, 0], [0, 1]]) # this is the field metric written in sympy notation

PyTransSetup.potential(V,nF,nP,False,G) # writes this potential and its derivatives into C++ file potential.h when run

PyTransSetup.compileName3("Angular",True) # this compiles a python module using the C++ code, including the edited potential.h file, called PyTransAngular
                                 # and places it in the location folder, ready for use

############################################################################################################################################

