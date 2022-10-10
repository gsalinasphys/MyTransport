import sys
from math import cos, log, pi

import sympy as sym
from gravipy.tensorial import *

location = "/home/gsalinas/GitHub/MyTransport/PyTransport" # this should be the location of the PyTransport folder
sys.path.append(location)  # we add this location to the python path

import PyTransSetup  # the above commands allows python to find the PyTransSetup module and import it

nF = 2  # number of fields
nP = 10  # number of parameters
f = sym.symarray('f',nF)   # an array representing the nF fields present for this model [phi, chi]
p = sym.symarray('p',nP)   # an array representing the nP parameters needed to define this model, format [Num, g, l, u, q, a0, a1, b1, p, V0]

L = (27 * pi / 4 * p[0] * p[1] * p[2] ** 4)**0.25
mu = ((2*pi)**5 * p[2]**6)**(-1)
T5 = mu / p[1]
gamma = 4 * pi**2 * p[2]**2 * p[-2] * p[4] * T5 * p[1]
rho = f[0] / 3 / p[3]

H = (L / 3 / p[3])**4 * (2 / rho**2 - 2 * sym.log(1/rho**2 + 1))
F = H / 9 * (f[0]**2 + 3*p[3]**2)**2 + (pi * p[2]**2 * p[4])**2
Phi = 5 / 72 * (81 * (9 * rho**2 - 2) * rho**2 + 162 * sym.log(9*(rho**2 + 1)) - 9 - 160*sym.log(10))
PhiH = p[5] * (2/rho**2 - 2*sym.log(1/rho**2 + 1)) + 2 * p[6] * (6 + 1/rho**2 - 2*(2+3*rho**2)*sym.log(1+1/rho**2))*sym.cos(f[1]) \
    + p[-3] / 2 * (2+3*rho**2) * sym.cos(f[1])

V = p[-1] + 4*pi*p[-2]*T5/H*(sym.sqrt(F)-p[2]**2*pi*p[4]*p[1])+gamma*(Phi+PhiH)   # this is the potential written in sympy notation
G = 4*pi*p[-2]*T5*sym.sqrt(F) * Matrix([[(f[0]**2+6*p[3]**2)/(f[0]**2+9*p[3]**2), 0], [0, 1/6*(f[0]**2+6*p[3]**2)]]) # this is the field metric written in sympy notation

PyTransSetup.potential(V,nF,nP,False,G) # writes this potential and its derivatives into C++ file potential.h when run

PyTransSetup.compileName3("D5",True) # this compiles a python module using the C++ code, including the edited potential.h file, called PyTransAngular
                                 # and places it in the location folder, ready for use

############################################################################################################################################

