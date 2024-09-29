import numpy as np
import utils
import sys
from scipy.linalg import expm

#Define class for Mash-rpmd
#It's a child-class of the rpmd parent class

import map_rpmd
import integrator

class mash_rpmd( map_rpmd.map_rpmd ):

    def __init__( self, nstates=1, nnuc=1, nbds=1, beta=1.0, mass=1.0, potype=None, potparams=None, mapR=None, mapP=None, mapSx=None, mapSy=None, mapSz=None, nucR=None, nucP=None, spinmap_bool=False):

        super().__init__( 'RP-MASH', nstates, nnuc, nbds, beta, mass, potype, potparams, mapR, mapP, nucR, nucP )

        if (spinmap_bool == True):

            if (nstates != 2):
                print('ERROR: The spin mapping variable is only for 2-state systems')
                exit()

            self.mapSx = mapSx
            self.mapSy = mapSy
            self.mapSz = mapSz

            self.spin_map_error_check()
        
        self.spin_map = spinmap_bool #Boolean that decide if we use spin mapping variables

        
    #####################################################################

    def get_timederivs( self ):
        #subroutine to calculate the time-derivatives of all position/momenta

        #update electronic Hamiltonian matrix
        self.potential.calc_Hel( self.nucR )

        #Calculate time-derivative of nuclear position
        dnucR = self.get_timederiv_nucR()

        #Calculate time-derivative of nuclear momenta
        dnucP = self.get_timederiv_nucP()

        #Calculate time-derivative of mapping position and momentum
        if (self.spin_map == False):
            dmapR = self.get_timederiv_mapR()
            dmapP = self.get_timederiv_mapP()

            return dnucR, dnucP, dmapR, dmapP
        
        else:
            dmapSx = self.get_timederiv_mapSx()
            dmapSy, dmapSz = self.get_timederiv_mapSyz()

            return dnucR, dnucP, dmapSx, dmapSy, dmapSz

    #####################################################################

    def get_timederiv_nucR( self ):
        #Subroutine to calculate the time-derivative of the nuclear positions for each bead

        return self.nucP / self.mass

    #####################################################################

    def get_timederiv_nucP( self, intRP_bool=True ):
        #Subroutine to calculate the time-derivative of the nuclear momenta for each bead

        #Force associated with harmonic springs between beads and the state-independent portion of the potential
        #This is dealt with in the parent class
        #If intRP_bool is False it does not calculate the contribution from the harmonic ring polymer springs

        if (self.nbds > 1):
            d_nucP = super().get_timederiv_nucP( intRP_bool )
        else:
            d_nucP = super().get_timederiv_nucP( intRP_bool = False )

        d_Vz = self.potential.get_bopes_derivs()[:,:,1]

        if (self.spin_map==False):
            #Calculate contribution from MMST term
            #XXX could maybe make this faster getting rid of double index in einsum
            d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapR, self.potential.d_Hel, self.mapR )
            d_nucP += -0.5 * np.einsum( 'in,ijnm,im -> ij', self.mapP, self.potential.d_Hel, self.mapP )

            #add the state-average potential
            if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
                d_nucP +=  0.5 * np.einsum( 'ijnn -> ij', self.potential.d_Hel )
        else:
            #The MASH nuclear force, note that Hel here are adiabatic surfaces
            d_nucP += - d_Vz * np.sign(self.mapSz) #Warning: be careful of the size of mapSz

        return d_nucP

   #####################################################################

    def get_timederiv_mapR( self ):
        #Subroutine to calculate the time-derivative of just the mapping position for each bead

        d_mapR =  np.einsum( 'inm,im->in', self.potential.Hel, self.mapP )

        return d_mapR

   #####################################################################

    def get_timederiv_mapP( self ):
        #Subroutine to calculate the time-derivative of just the mapping momentum for each bead

        d_mapP = -np.einsum( 'inm,im->in', self.potential.Hel, self.mapR )

        return d_mapP

   #####################################################################

    def get_2nd_timederiv_mapR( self, d_mapP ):
        #Subroutine to calculate the second time-derivative of just the mapping positions for each bead
        #This assumes that the nuclei are fixed - used in vv style integrators

        d2_mapR = np.einsum( 'inm,im->in', self.potential.Hel, d_mapP )

        return d2_mapR

   #####################################################################

    def get_timederiv_mapSx( self ):

        Vz = self.potential.get_bopes()[:,1]
        NAC = self.potential.calc_NAC()

        d_mapSx = 2 * np.sum(NAC * self.nucP / self.mass, axis = 1) * self.mapSz - 2 * Vz * self.mapSy

        return d_mapSx

   #####################################################################

    def get_timederiv_mapSyz(self):

        Vz = self.potential.get_bopes()[:,1]
        NAC = self.potential.calc_NAC()

        d_mapSy = 2 * np.sum(NAC * self.nucP / self.mass, axis = 1) * self.mapSx
        d_mapSz = -2 * Vz * self.mapSx

        return d_mapSy, d_mapSz

   #####################################################################
    
    def get_2nd_timederiv_mapSx( self, d_mapSy, d_mapSz):

        d2_mapSx = 2 * np.sum(self.NAC * self.nucP, axis = 1) / self.mass * d_mapSz - 2 * self.potential.Hel * d_mapSy

        return d2_mapSx

   #####################################################################

    def get_PE( self ):
        #Subroutine to calculate potential energy associated with mapping variables and nuclear position

        #Internal ring-polymer modes, 0 if there is only one bead (i.e., LSC-IVR)
        if self.nbds > 1:
            engpe = self.potential.calc_rp_harm_eng( self.nucR, self.beta_p, self.mass )
        else:
            engpe = 0

        #State independent term
        engpe += self.potential.calc_state_indep_eng( self.nucR )

        #Update electronic Hamiltonian matrix
        self.potential.calc_Hel(self.nucR)

        if(self.spin_map==False):
            #MMST Term
            engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapR, self.potential.Hel, self.mapR ) )
            engpe += 0.5 * np.sum( np.einsum( 'in,inm,im -> i', self.mapP, self.potential.Hel, self.mapP ) )

            if (self.potype != 'harm_lin_cpl_symmetrized' or self.potype != 'harm_lin_cpl_sym_2'):
                engpe += -0.5 * np.sum( np.einsum( 'inn -> i', self.potential.Hel ) )

        else:
            engpe += np.sum(self.potential.Hel * np.sign(self.mapSz))
        return engpe

    #####################################################################

    def init_spin_variables(self):
        pass

    def spin_map_error_check(self):

        if( self.mapSx is not None and self.mapSx.shape != (self.nbds,) ):
            print('ERROR: Size of spin mapping variable Sx doesnt match bead number')
            exit()

        if( self.mapSy is not None and self.mapSy.shape != (self.nbds,) ):
            print('ERROR: Size of spin mapping variable Sy doesnt match bead number')
            exit()

        if( self.mapSz is not None and self.mapSz.shape != (self.nbds,) ):
            print('ERROR: Size of spin mapping variable Sz doesnt match bead number')
            exit()

    #####################################################################

    def get_sampling_eng(self):
        return None

    #####################################################################

    def print_data( self, step ):
        return None


