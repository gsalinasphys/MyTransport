import os
import pickle
import sys

import numpy as np
import sympy
from matplotlib import pyplot as plt
from sympy import Matrix, cos, sin, symbols
from sympy.diffgeom import CoordSystem, Manifold, Patch
from sympy.diffgeom import TensorProduct as TP
from sympy.diffgeom import metric_to_Christoffel_2nd, twoform_to_matrix
from sympy.utilities import lambdify

location = "/home/gsalinas/GitHub/MyTransport"
sys.path.append(location + '/PyTransport')
location_model = location + '/Examples/AngularPolar'

import PyTransSetup

manifold = Manifold('M', 2)
patch = Patch('P', manifold)

fields = symbols('r theta', real=True, nonnegative=True)
r, theta = fields
params = symbols('alpha R m_phi', real=True, nonnegative=True)
alpha, R, m_phi = params
nF, nP = len(fields), len(params)
f = sympy.symarray('f', nF)
p = sympy.symarray('p', nP)
dict_fields = dict(zip(fields, f))
dict_params = dict(zip(params, p))
dict_combined = {**dict_params, **dict_fields}

coord_syst = CoordSystem('Pol', patch, fields)
fr, ftheta = coord_syst.base_scalars()
e_r, e_theta = coord_syst.base_vectors()
dr, dtheta = coord_syst.base_oneforms()

metric_polar = 6 * alpha / (1 - fr**2)**2 * (TP(dr, dr) + fr**2*TP(dtheta, dtheta))
G = twoform_to_matrix(metric_polar).simplify()
Gammas = metric_to_Christoffel_2nd(metric_polar).simplify()

V = alpha/2 * m_phi**2 * r**2 * (cos(theta)**2 + R*sin(theta)**2)

with open(location_model + "/ModelSetup/G.txt", "wb") as f:
    pickle.dump(G, f)
with open(location_model + "/ModelSetup/Gammas.txt", "wb") as f:
    pickle.dump(Gammas, f)
with open(location_model + "/ModelSetup/V.txt", "wb") as f:
    pickle.dump(V, f)

with open(location_model + '/ModelSetup/info.txt', 'w') as f:
    f.write('Fields: ' + str(fields) + '\n')
    f.write('Parameters: ' + str(params) + '\n\n')
    f.write('Potential:\n\nV = ' + str(V) + '\n\n')
    f.write('Field space metric:\n\nG = ' + str(G))

V = V.subs(dict_combined)
G = G.subs(dict_combined)
PyTransSetup.potential(V, nF, nP, False, G)

PyTransSetup.compileName3("AngularPolar", True) 
