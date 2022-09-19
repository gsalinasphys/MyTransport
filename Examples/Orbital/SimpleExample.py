import os
import sys

import numpy as np
import sympy as sym
from gravipy.tensorial import *
from matplotlib import pyplot as plt
from pylab import *

# This file contains simple examples of using the Orbital 
# It assumes the OrbitalSetup file has been run to install a orbital inflation version of PyTransport

locdir = os.getcwd() + '/output'
location = "/home/gsalinas/GitHub/PyTransport/PyTransport" # this should be the location of the PyTransport folder folder
sys.path.append(location) # sets up python path to give access to PyTransSetup
import PyTransSetup

PyTransSetup.pathSet()  # this adds the other paths that PyTransport uses to the python path


###########################################################################################################################################


import PyTransOrbital as PyT
import PyTransScripts as PyS

nF = 2  # number of fields
nP = 5  # number of parameters
f = sym.symarray('f',nF)   # an array representing the nF fields present for this model
p = sym.symarray('p',nP)   # an array representing the nP parameters needed to define this model, in the form [R0, alpha, lambda, A, rho0]

Vexpr = 3 * p[3]**2.0 * (f[0]**2.-2/(3*sym.exp(2*f[1]/p[0]))) * (1+p[2]/2.*(f[1]-p[4])**2.+p[1]/6*(f[1]-p[4])**3)**2. \
    - 2 * p[3]**2. * f[0]**2. * (p[2]*(f[1]-p[4])+p[1]/2*(f[1]-p[4])**2.)**2
G = sym.Matrix( [[sym.exp(2*f[1]/p[0]), 0], [0, 1] ] ) # this is the field metric written in sympy notation
Ga = PyTransSetup.fieldmetric(G, nF, nP)[1]


########################### Set some field values and field derivatives in cosmic time ####################################################


fields = np.array([30.,1.]) # initial values of the fields
nP = PyT.nP()   # the .nP function gets the number of parameters needed for the potential (can be used as a useful cross check)
pval = np.array([16., 1., 0.1/6, 6.66e-7, fields[1]])    # in the form [R0, alpha, lambda, A, rho0]
nF = PyT.nF() # use number of fields routine to get number of fields (can be used as a useful cross check)

V = PyT.V(fields,pval) # calculate potential from some initial conditions
dV = PyT.dV(fields,pval) # calculate derivatives of potential

initial = np.concatenate((fields, np.array([0.,0.]))) # sets an array containing field values and their derivatives in cosmic time 

Vr = sym.diff(sym.log(Vexpr), 'f_1').subs([('p_0', pval[0]),('p_1', pval[1]),('p_2', pval[2]),('p_3', pval[3]), ('p_4', pval[4]), ('f_1', pval[4])])
Vtt = sym.diff(sym.log(Vexpr), 'f_0', 'f_0').subs([('p_0', pval[0]),('p_1', pval[1]),('p_2', pval[2]),('p_3', pval[3]), ('p_4', pval[4]), ('f_1', pval[4])])


################################## Run and plot the background fiducial run ################################################################
Nstart = 0.0
Nend = 200.
t=np.linspace(Nstart, Nend, 1000)

tols = np.array([10**-10,10**-10])  # Absolute and relative tolerances
back = PyT.backEvolve(t, initial, pval, tols, True)

plt.plot(back[:,0], back[:,1], 'g', label='$\\theta$')
plt.plot(back[:,0], back[:,2], 'r', label='$\\rho$')
plt.title(r'Background evolution',fontsize=15)
plt.legend(fontsize=16)
plt.ylabel(r'Fields', fontsize=20)
plt.xlabel(r'$N$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(locdir+"/Orbital1.png")
plt.close()

plt.plot(back[:,0], back[:,3], 'g', label='$\\theta$')
plt.plot(back[:,0], back[:,4], 'r', label='$\\rho$')
plt.title(r'Background evolution',fontsize=15)
plt.legend(fontsize=16)
plt.ylabel(r'Field velocities', fontsize=20)
plt.xlabel(r'$N$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(locdir+"/Orbital2.png")
plt.close()

############################################################################################################################################

# Find Hubble rate and epsilon parameter
Hs = np.array([PyT.H(elem, pval) for elem in back[:, 1:]])

dt = t[1]-t[0]
epsilon = -np.gradient(Hs, dt)/Hs

plt.plot(t[:len(Hs)], Hs)
plt.ylabel(r'Hubble rate', fontsize=20)
plt.xlabel(r'$N$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(locdir+"/Orbital3.png")
plt.close()

plt.plot(t[:len(epsilon)], epsilon)
plt.ylabel('$\\epsilon$', fontsize=20)
plt.xlabel(r'$N$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(locdir+"/Orbital4.png")
plt.close()

ddphi = np.array([np.gradient(back[:, ii], dt) for ii in range(3, 5)])
ddphi[0] = ddphi[0]/Hs
ddphi[0] = ddphi[0] + epsilon*back[:, 3]/Hs
ddphi[1] = ddphi[1]/Hs
G112 = Ga(-1, 1, 2).subs('p_0', pval[0])
G211 = Ga(-2, 1, 1).subs([('p_0', pval[0]), ('f_1', pval[4])])
eta = ddphi + np.array([2*G112*back[:, 3]*back[:, 4], G211*back[:, 3]**2])/Hs**2

frho = np.exp(2*back[:,2]/pval[0])
# detapllrho = -2/pval[0]*(back[:,3]**2/(Hs**2*pval[0]) + eta[0]/frho)
Vrs = np.array([Vr.subs('f_0', theta) for theta in back[:,1]])
Vtts = np.array([Vtt.subs('f_0', theta) for theta in back[:,1]])
Mpp = Vtts - G211*Vrs + frho/pval[0]*eta[1]/(3-epsilon)
# print(Mpp/frho)

Mpp2 = -eta[0]/np.sqrt(2*epsilon*frho)
# print(Mpp2)

# plt.plot(t[200:], Mpp[200:])
# plt.tight_layout()
# plt.show()
# plt.close()

plt.plot(t[:len(eta[0])], eta[0]/np.sqrt(2*epsilon*np.exp(2*back[:,2]/pval[0])))
plt.ylabel('$\\eta_\\parallel$', fontsize=20)
plt.xlabel(r'$N$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(locdir+"/Orbital5.png")
plt.close()

plt.plot(t[:len(eta[1])], eta[1]/np.sqrt(2*epsilon))
plt.ylabel('$\\eta_\\perp$', fontsize=20)
plt.xlabel(r'$N$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(locdir+"/Orbital6.png")
plt.close()


############################################################################################################################################


# Set a pivot scale which exits after certain time using the background run -- a spline
# is used to find field and field velocity values after Nexit number of e-folds, this gives H, and 
# then k=aH gives the k pivot scale in this example we treat this scale as k_t

NExit = 15.0
k = PyS.kexitN(NExit, back, pval, PyT) 

# other scales can then be defined wrt to k


################################# Example 2pt run ##########################################################################################

NB = 6.0
Nstart, backExitMinus = PyS.ICsBE(NB, k, back, pval, PyT) # find conditions for 6 e-folds before horizon crossing of k mode


tsig=np.linspace(Nstart,back[-1,0], 1000)  # array of times (e-folds) at which output is returned -- initial time should correspond to initial field
                                           # and velocity values which will be fed in to the functions which evolve correlations


# run the sigma routine to calc and plot the evolution of power spectrum value for this k -- can be
# repeated to build up the spectrum, here we run twice to get an crude estimate for ns
twoPt = PyT.sigEvolve(tsig, k, backExitMinus,pval,tols, True) # puts information about the two point fuction in twoPt array
zz1=twoPt[:,1] # the second column is the 2pt of zeta
sigma = twoPt[:,1+1+2*nF:] # the last 2nF* 2nF columns correspond to the evolution of the sigma matrix
zz1a=zz1[-1] # the value of the power spectrum for this k value at the end of the run

twoPt=PyT.sigEvolve(tsig, k+.1*k, backExitMinus, pval,tols, True)
zz2=twoPt[:,1]
zz2a=zz2[-1]
n_s = (np.log(zz2a)-np.log(zz1a))/(np.log(k+.1*k)-np.log(k))+4.0

pairs = [(0,0), (0,1), (1,1)]
labels = ['$\\Sigma^{\\theta\\theta}$', '$\\Sigma^{\\theta\\rho}$', '$\\Sigma^{\\rho\\rho}$']
for ii, pair in enumerate(pairs):
    plt.plot(twoPt[:,0], np.abs(sigma[:,pair[0] + 2*nF*pair[1]]), label=labels[ii])
plt.title(r'$\Sigma$ evolution',fontsize=16)
plt.legend(fontsize=16)
plt.ylabel(r'Aboslute 2pt field correlations', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig(locdir+"/Orbital7.png")
plt.close()

plt.plot(tsig, zz1[:])
title(r'$P_\zeta$ evolution',fontsize=16);
plt.ylabel(r'$P_\zeta(k)$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig(locdir+"/Orbital8.png")
plt.close()


###################################### Example bispectrum run ##############################################################################


# set three scales in FLS manner (using alpha, beta notation)
alpha = 0.
beta = 1./3.
k1 = k/2 - beta*k/2. ; k2 = k/4*(1+alpha+beta) ; k3 = k/4*(1-alpha+beta)

# find initial conditions for 6 e-folds before the smallest k (which exits the horizon first) crosses the horizon
kM = np.min(np.array([k1,k2,k3]))
Nstart, backExitMinus = PyS.ICsBM(NB, kM, back, pval, PyT)

# run the three point evolution for this triangle
talp=np.linspace(Nstart,back[-1,0], 1000)
threePt = PyT.alphaEvolve(talp,k1,k2,k3, backExitMinus,pval,tols,True) # all data from three point run goes into threePt array
alpha= threePt[:,1+4+2*nF+6*2*nF*2*nF:]        # this now contains the 3pt of the fields and field derivative perturbations
zzz= threePt[:,1:5] # this contains the evolution of two point of zeta for each k mode involved and the 3pt of zeta

for ii in range(0,2):
    for jj in range(ii,2):
        for kk in range(jj,2):
            plt.plot(talp, np.abs(alpha[:,ii + 2*nF*jj + 2*nF*2*nF*kk]), label='$\\alpha$'+'('+str(ii)+str(jj)+str(kk)+')')
plt.title(r'$\alpha$ evolution',fontsize=16)
plt.legend(fontsize=16)
plt.ylabel(r'Absolute 3pt field correlations', fontsize=20)
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig(locdir+"/Orbital9.png")
plt.close()

fnl = 5.0/6.0*zzz[:,3]/(zzz[:,1]*zzz[:,2]  + zzz[:,0]*zzz[:,1] + zzz[:,0]*zzz[:,2])
plt.plot(talp[200:], fnl[200:],'r')
plt.title(r'$f_{NL}$ evolution',fontsize=15)
plt.ylabel(r'$f_{NL}$', fontsize=20); 
plt.xlabel(r'$N$', fontsize=20)
plt.tight_layout()
plt.savefig(locdir+"/Orbital10.png")
plt.close()
