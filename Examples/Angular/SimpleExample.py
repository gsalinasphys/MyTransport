import sys
from math import cos, pi, sin, sqrt

import numpy as np
import sympy as sym
from matplotlib import pyplot as plt

location = "/home/gsalinas/GitHub/MyTransport/PyTransport" # This should be the location of the PyTransport folder folder
sys.path.append(location) # Sets up python path to give access to PyTransSetup

import PyTransSetup

PyTransSetup.pathSet()  # This adds the other paths that PyTransport uses to the python path

import PyTransAngular as PyT
import PyTransScripts as PyS

# nF = 2  # number of fields
# nP = 2  # number of parameters
# f = sym.symarray('f',nF)   # an array representing the nF fields present for this model
# p = sym.symarray('p',nP)   # an array representing the nP parameters needed to define this model, format [alpha, R]

# V = p[0]/2 * p[2]** 2* (f[0]**2 + p[1]*f[1]**2)   # this is the potential written in sympy notation
# G = 6*p[0]/(1-f[0]**2-f[1]**2)**2 * sym.Matrix([[1, 0], [0, 1]]) # this is the field metric written in sympy notation

# Gamma = PyTransSetup.fieldmetric(G, nF, nP)[1]
# print(Gamma(-1, 1, 2))

########################### Set some field values and field derivatives in cosmic time ####################################################

nF=PyT.nF()
nP=PyT.nP()

pval = np.array([1/600, 9, 2e-5]) # Parameter values, format [alpha, R, mphi]
r0, theta0 = 0.99, pi/4
fields = r0 * np.array([cos(theta0), sin(theta0)]) # Initial values of the fields [phi, chi]

V = PyT.V(fields, pval) # Calculate potential from some initial conditions
dV = PyT.dV(fields, pval) # Calculate derivatives of potential

phidot0 = np.zeros(2) # set initial conditions to be in slow roll
initial = np.concatenate((fields, phidot0)) # Sets an array containing field values and there derivative in cosmic time 

################################## Run and plot the background fiducial run ################################################################
print('Starting background evolution.')

Nstart, Nend = 0., 150
t = np.linspace(Nstart, Nend, 100_000)

tols = np.array([10**-6, 10**-6])
back = PyT.backEvolve(t, initial, pval, tols, True)

# def lastNe(back: np.ndarray, N: int = 60) -> np.ndarray:
#     assert back[-1, 0] - back[0, 0] > N, 'Needs more than N e-folds'
    
#     back = back[np.where(back[-1, 0] - back[:, 0] < N)[0], :]
#     back[:, 0] -= np.min(back[:, 0])
#     return back

# Nefolds = 100
# back = lastNe(back, Nefolds)

Nend = back[-1,0]
print('Number of e-folds: ', Nend)

rs = np.sqrt(back[:, 1]**2 + back[:, 2]**2)
psis = sqrt(6*pval[0]) * np.arctanh(rs)
thetas = np.arctan(back[:,2]/back[:,1])
plt.plot(psis * np.cos(thetas), psis * np.sin(thetas))
plt.xlabel(r'$\psi \cos(\theta)$')
plt.ylabel(r'$\psi \sin(\theta)$')
plt.tight_layout()
plt.savefig('Examples/Angular/Background.png')
plt.clf()    

# Find Hubble rate
Hs = np.array([PyT.H(elem, pval) for elem in back[:, 1:]])
plt.plot(back[:, 0], Hs)
plt.title('Hubble parameter')
plt.xlabel(r'$N$', fontsize=16)
plt.ylabel(r'$H$', fontsize=16)
plt.tight_layout()
plt.savefig('Examples/Angular/Hubble.png')
plt.clf()
print('Average H: ', np.mean(Hs))

# Find epsilon
dt = t[1] - t[0]
epsilon = -np.gradient(Hs, dt)/Hs
plt.plot(back[:, 0], epsilon)
plt.title('Epsilon parameter')
plt.xlabel(r'$N$', fontsize=16)
plt.ylabel(r'$\epsilon$', fontsize=16)
plt.tight_layout()
plt.savefig('Examples/Angular/Epsilon.png')
plt.clf()
print("Initial epsilon: ", epsilon[0])

# # Turn rate
# def turn_rate(hat_sigma: np.ndarray, hat_thetas: np.ndarray, dt: float) -> float:
#     return np.sum(hat_rhos * np.gradient(hat_thetas, dt, axis=0), axis=1)

# turn_rate = 

# plt.plot(back[:, 0], turn_rate(hat_rhos, hat_thetas, dt))
# plt.title('Turn rate')
# plt.xlabel(r'$N$', fontsize=16)
# plt.ylabel(r'$\omega$', fontsize=16)
# plt.savefig('Examples/Angular/TurnRate.png')
# plt.clf()

# plt.plot(back[:, 0], np.gradient(turn_rate(hat_rhos, hat_thetas, dt), dt) / turn_rate(hat_rhos, hat_thetas, dt))
# plt.title('Turn rate')
# plt.xlabel(r'$N$', fontsize=16)
# plt.ylabel(r'$\omega^\prime / \omega$', fontsize=16)
# plt.savefig('Examples/Angular/DTurnRate.png')
# plt.clf()

# print('Done with background evolution, starting 2-pt correlation calculation.')

############################################################################################################################################
# set a pivot scale which exits after certain time using the background run -- a spline
# is used to find field and field velocity values after Nexit number of e-folds, this gives H, and 
# then k=aH gives the k pivot scale
# in this example we treat this scale as k_t

NExit = Nend - 55
print('Horizon exit at: ', NExit)
k = PyS.kexitN(NExit, back, pval, PyT) 

# other scales can then be defined wrt to k


# ################################# example 2pt run ##########################################################################################

NB = 6.0
Nstart, backExitMinus = PyS.ICsBE(NB, k, back, pval, PyT) # find conditions for NB e-folds before horizon crossing of k mode
print(f"2-pt calculation starts at: {Nstart} e-folds")

tsig = np.linspace(Nstart, Nend, 1_000)  # array of times (e-folds) at which output is returned -- initial time should correspond to initial field
                                     # and velocity values which will be fed in to the functions which evolve correlations


# run the sigma routine to calc and plot the evolution of power spectrum value for this k -- can be
# repeated to build up the spectrum, here we run twice to get an crude estimate for ns
twoPt = PyT.sigEvolve(tsig, k, backExitMinus, pval, tols, True) # puts information about the two point fuction in twoPt array
zz1 = twoPt[:, 1] # the second column is the 2pt of zeta
sigma = twoPt[:, 1+1+2*nF:] # the last 2nF* 2nF columns correspond to the evolution of the sigma matrix
zz1a = zz1[-1] # the value of the power spectrum for this k value at the end of the run

twoPt = PyT.sigEvolve(tsig, k+.1*k, backExitMinus, pval, tols, True)
zz2 = twoPt[:,1]
zz2a = zz2[-1]
n_s = (np.log(zz2a)-np.log(zz1a))/(np.log(k+.1*k)-np.log(k))+4.0
print(f'n_s: {n_s}')

pairs = [(0,0), (0,1), (1,1)]
labels = ['$P^{\\phi\\phi}_\\phi$', '$P^{\\phi\\chi}_\\phi$', '$P^{\\chi\\chi}_\\phi$']
for ii, pair in enumerate(pairs):
    plt.plot(twoPt[:, 0], np.abs(sigma[:, pair[0] + 2*nF*pair[1]]), label=labels[ii])
plt.title(r'$\Sigma$ evolution',fontsize=16)
plt.legend(fontsize=16)
plt.ylabel(r'Aboslute 2pt field correlations', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.axvline(NExit, c='k', linestyle='--')
plt.tight_layout()
plt.savefig('Examples/Angular/Field2pt.png')
plt.clf()

sig_rr = np.abs(sigma[:, 0])
sig_rth = np.abs(sigma[:, 2*nF])
sig_thth = np.abs(sigma[:, 2*nF + 1])
i_exit = np.argmin(np.abs(twoPt[:, 0] - NExit))
print('At horizon crossing:')
print('$P^{\\phi\\phi}_\\phi$: ', sig_rr[i_exit])
print('$P^{\\phi\\chi}_\\phi$: ', sig_rth[i_exit])
print('$P^{\\chi\\chi}_\\phi$: ', sig_thth[i_exit])

Pzeta = zz1[:] * k**3 / 2 / np.pi**2
plt.plot(tsig, Pzeta)
plt.axvline(NExit, c='k', linestyle='--')
plt.title(r'$P_\zeta$ evolution',fontsize=16);
plt.ylabel(r'$P_\zeta(k)$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig('Examples/Angular/PowerSpectrum.png')
plt.clf()
print(f'Power spectrum: {zz1[:][-1] * k**3 / 2 / np.pi**2} at k = {k}')
print(f'Power spectrum at horizon crossing: {zz1[:][i_exit] * k**3 / 2 / np.pi**2}')


print('Done with 2-pt, starting bispectrum calculation.')

###################################### example bispectrum run ##############################################################################

# # set three scales in FLS manner (using alpha, beta notation)
# alpha = 0.
# beta = 1/3.

# k1 = k/2 - beta*k/2.
# k2 = k/4*(1+alpha+beta)
# k3 = k/4*(1-alpha+beta)

k1, k2, k3 = k, k, k

# find initial conditions for NB e-folds before the smallest k (which exits the horizon first) crosses the horizon
kM = np.min(np.array([k1, k2, k3]))
Nstart, backExitMinus = PyS.ICsBM(NB, kM, back, pval, PyT)
print(f"3-pt calculation starts at: {Nstart} e-folds")

# run the three point evolution for this triangle
talp = np.linspace(Nstart, Nend, 10_000)
threePt = PyT.alphaEvolve(talp, k1, k2, k3, backExitMinus, pval, tols, True) # all data from three point run goes into threePt array
alpha = threePt[:,1+4+2*nF+6*2*nF*2*nF:]        # this now contains the 3pt of the fields and field derivative pertruabtions
zzz = threePt[:,1:5] # this contains the evolution of two point of zeta for each k mode involved and the 3pt of zeta

for ii in range(0,2):
    for jj in range(0,2):
        for kk in range(0,2):
            plt.plot(talp, np.abs(alpha[:,ii + 2*nF*jj + 2*nF*2*nF*kk]))
plt.title(r'$\alpha$ evolution',fontsize=15)
plt.ylabel(r'Absolute 3pt field correlations', fontsize=20)
plt.xlabel(r'$N$', fontsize=15)
plt.yscale('log')
plt.tight_layout()
plt.savefig("Examples/Angular/Field3pt.png")
plt.clf()

fnl = 5.0/6.0*zzz[:,3]/(zzz[:,1]*zzz[:,2]  + zzz[:,0]*zzz[:,1] + zzz[:,0]*zzz[:,2])
plt.plot(talp, fnl,'r')
plt.title(r'$f_{NL}$ evolution',fontsize=15)
plt.ylabel(r'$f_{NL}$', fontsize=20)
plt.xlabel(r'$N$', fontsize=15)
plt.tight_layout()
plt.savefig("Examples/Angular/fNL.png")
plt.clf()

print(f'fNL: {np.median(fnl[len(talp)//5:])}')

print('Done with bispectrum.')
