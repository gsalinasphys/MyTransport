####################################### Setup file for the Quartic Axion non-canonical Model ######################################################

import sympy as sym    # we import the sympy package
import math            # we import the math package (not used here, but has useful constants such math.pi which might be needed in other cases)    
import sys             # import the sys module used below
from pylab import *
from gravipy.tensorial import *

############################################################################################################################################

# if using an integrated environment we recommend restarting the python console after running this script to make sure updates are found 

location = "/home/jwr/Code/PyTransport/" # this should be the location of the PyTransport folder 
sys.path.append(location)  # we add this location to the python path

import PyTransSetup  # the above commands allows python to find the PyTransSetup module and import it

############################################################################################################################################

nF=2
nP=4

f=sym.symarray('f',nF)
p=sym.symarray('p',nP)
G=Matrix( [[p[3]**2.0, 0], [0, p[3]**2.0*sym.sin(f[0])**2.0] ] )
V= 1./4. * p[0] * f[0]**4 + p[2] * (1-sym.cos(2*math.pi * f[1] / p[1]))

PyTransSetup.potential(V,nF,nP,False,G) # writes this potential into c file when run

PyTransSetup.compileName("QuartAxNC",True) 
