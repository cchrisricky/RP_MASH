import numpy as np
import sys
import os
sys.path.append('RP_MASH')
import mash_rpmd
import utils
import linecache

mash_ = mash_rpmd.mash_rpmd(nstates=2, nnuc=3, nbds=1, beta=16, mass=np.array([1, 1, 1]), potype='harm_const_cpl'
                            , potparams=[np.array([1,0.5,0.5]), np.array([[0.2,-0.2],[0.1,-0.1],[0.1,-0.1]]), np.array([0,0]), np.array([0.5])]
                            , mapR=None, mapP=None, mapSx=np.array([-0.5]), mapSy=np.array([0.5]), mapSz=np.array([0.5])
                            , nucR=np.array([[-0.1,-0.2,0.]]), nucP=np.array([[0.,0.,0.]]), spinmap_bool=True)

mash_.potential.calc_Hel(mash_.nucR)
mash_.potential.calc_Hel_deriv(mash_.nucR)

# printing various attributes for debuging/testing
print(mash_.potential.calc_NAC())
print(mash_.potential.Hel)
print(mash_.potential.d_Hel)
print(mash_.potential.get_bopes_derivs())
print(mash_.get_timederiv_mapSx())
print(mash_.get_timederiv_nucP())
print(mash_.potential.get_bopes()[:,1])

