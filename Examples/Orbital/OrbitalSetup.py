import sys

import sympy as sym
from gravipy.tensorial import *  # we import the gravipy.tensorial package to write the field metric

location = "/home/gsalinas/GitHub/PyTransport/PyTransport" # this should be the location of the PyTransport folder
sys.path.append(location)  # we add this location to the python path
import PyTransSetup  # the above commands allows python to find the PyTransSetup module and import it

############################################################################################################################################

nF = 2  # number of fields
nP = 5  # number of parameters
f = sym.symarray('f',nF)   # an array representing the nF fields present for this model
p = sym.symarray('p',nP)   # an array representing the nP parameters needed to define this model, in the form [R0, alpha, lambda, A, rho0]

V = 3 * p[3]**2.0 * (f[0]**2.-2/(3*sym.exp(2*f[1]/p[0]))) * (1+p[2]/2.*(f[1]-p[4])**2.+p[1]/6*(f[1]-p[4])**3)**2. \
    - 2 * p[3]**2. * f[0]**2. * (p[2]*(f[1]-p[4])+p[1]/2*(f[1]-p[4])**2.)**2
    
G=sym.Matrix([[sym.exp(2*f[1]/p[0]), 0.], [0., 1.]]) # this is the field metric written in sympy notation

PyTransSetup.potential(V,nF,nP,False,G) # writes this potential and its derivatives into C++ file potential.h when run

PyTransSetup.compileName3("Orbital",True) # this compiles a python module using the C++ code, including the edited potential.h file, called Orbital
                                          # and places it in the location folder, ready for use

