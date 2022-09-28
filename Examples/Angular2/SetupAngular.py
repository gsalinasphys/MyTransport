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
location_model = location + '/Examples/Angular2'

import PyTransSetup

manifold = Manifold('M', 2)
patch = Patch('P', manifold)

fields = symbols('phi chi', real=True, nonnegative=True)
phi, chi = fields
params = symbols('alpha R m_phi', real=True, nonnegative=True)
alpha, R, m_phi = params
nF, nP = len(fields), len(params)
f = sympy.symarray('f', nF)
p = sympy.symarray('p', nP)
dict_fields = dict(zip(fields, f))
dict_params = dict(zip(params, p))
dict_combined = {**dict_params, **dict_fields}

coord_syst = CoordSystem('Cart', patch, fields)
fphi, fchi = coord_syst.base_scalars()
e_phi, e_chi = coord_syst.base_vectors()
dphi, dchi = coord_syst.base_oneforms()

metric = 6 * alpha / (1 - fphi**2 - fchi**2)**2 * (TP(dphi, dphi) + TP(dchi, dchi))
G = twoform_to_matrix(metric).simplify().subs([(fphi, phi), (fchi, chi)])
Gammas = metric_to_Christoffel_2nd(metric).simplify().subs([(fphi, phi), (fchi, chi)])

V = alpha/2 * m_phi**2 * (phi**2 + R*chi**2)

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

PyTransSetup.compileName3("Angular", True) 
