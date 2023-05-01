# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/physics/multiphaseWLMA/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the WLMA implementation in Quail.
#
# ------------------------------------------------------------------------ #
from enum import Enum, auto
import logging
import numpy as np
import scipy.integrate
import scipy.special as sp

# Import submodule
import physics.multiphaseWLMA.iapws95_light.mixtureWLMA as mixtureWLMA

import errors
import general

from physics.base.data import (BCBase, FcnBase, BCWeakRiemann, BCWeakPrescribed,
				SourceBase, ConvNumFluxBase)
import physics.multiphasevpT.atomics as atomics

class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.
	'''
	IsothermalAtmosphere = auto()


class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions.
	'''
	''' BCs specific to multiphaseWLMA (see also
	physics.multiphasevpT.functions). '''
	pass
	# SlipWall = auto()
	LinearizedImpedance2D = auto()
	# MultiphasevpT1D1D = auto()
	# MultiphasevpT2D1D = auto()
	# MultiphasevpT2D2D = auto()
	# MultiphasevpT2D1DCylindrical = auto()
	# MultiphasevpT2D2DCylindrical = auto()
	# Not implemented (could use lumped magma chamber model for example)

class SourceType(Enum):
	''' SourceTypes specific to multiphaseWLMA (see also
	physics.multiphasevpT.functions). '''
	pass
	# FrictionVolFracVariableMu = auto()
	# FrictionVolFracConstMu = auto()
	# GravitySource = auto()
	# ExsolutionSource = auto()
	# FragmentationTimescaleSource = auto()
	# WaterInflowSource = auto()
	# CylindricalGeometricSource = auto()


class ConvNumFluxType(Enum):
	'''
	Enum class that stores the types of convective numerical fluxes. These
	numerical fluxes are specific to the available equation sets.
	'''
	LaxFriedrichs = auto()


'''
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
'''

class IsothermalAtmosphere(FcnBase):
	'''
	Isothermal air atmosphere as an initial condition.
	'''

	def __init__(self,T:float=300., p_h0:float=1e3*1600*9.8,
		h0:float=-400.0, hmax:float=1000.0, gravity:float=9.8,
		massFracWv:float=1.0-1e-7, massFracM:float=1e-7, tracer_frac:float=1e-7):
		''' Set atmosphere temperature, pressure, and location of pressure.
		Pressure distribution is computed as hydrostatic profile with p = p_h0
		at elevation h0.
		'''
		self.T = T
		self.p_h0 = p_h0
		self.h0 = h0
		self.hmax = hmax
		self.gravity = gravity
		self.massFracWv = massFracWv
		self.massFracM = massFracM
		# Allocate pressure interpolant (filled when self.get_state is called
		# because physics object is required)
		self.pressure_interpolant = None
		# Set numerical fraction for essentially inert fields
		self.tracer_frac = tracer_frac

	def get_state(self, physics, x, t):
		''' Computes the pressure in an isothermal atmosphere for an ideal gas
		mixture with air and water vapour. Trace amounts of magma phase are added
		and the pressure profile is iteratively corrected. '''

		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		# Compute mole fractions times R_univ
		nA_R = (1.0 - self.massFracWv) * physics.Gas[0]["R"]
		nWv_R = self.massFracWv * physics.Gas[1]["R"]
		# Compute mixture gas constant (gas constant per average molar mass)
		R = nA_R + nWv_R
		# Compute mole fraction of gas (=volume fraction of gas part)
		nA = nA_R / R
		nWv = nWv_R / R
		# Cache mole fractions and properties
		self.nA = nA
		self.nWv = nWv
		self.Gas = physics.Gas
		self.Liquid = physics.Liquid
		# Compute scale height (constant)
		hs = R*self.T/self.gravity

		''' Compute hydrostatic pressure as initial value problem (IVP) '''
		# Compare this to
		#   plt.plot(np.unique(x[...,1]),
		#     np.exp(-(np.unique(x[...,1])-x[...,1].min())/hs)*1e5, '-')

		# Compute (nearly exponential) pressure p as a function of y
		self.y = np.array([1.0-self.massFracWv-self.massFracM,
			self.massFracWv, self.massFracM])
		# TODO: generalize
		# Renormalize out air
		if self.y[0] < 0:
			self.y[0] = 0
			self.y /= self.y.sum()

		# TODO: currently pure water, then add magma trace
		eval_pts = np.unique(x[:,:,1:2])
		# Specific volume as function of p, T
		mixture_spec_vol = lambda p, T: \
			self.y[0] * physics.Gas[0]["R"] * T / p \
			+ self.y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
			+ self.y[2] / (physics.Liquid["rho0"] 
				* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))
		
		# Evaluate IVP solution
		soln = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / mixture_spec_vol(p, self.T),
			[self.h0, self.hmax],
			[self.p_h0],
			t_eval=eval_pts,
			dense_output=True)
		# Cache the pressure interpolant
		self.pressure_interpolant = soln.sol
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p = np.reshape(self.pressure_interpolant(x[...,1].ravel()), x[...,1].shape)
		rho = 1.0 / np.array([mixture_spec_vol(_p, self.T) for _p in p])
		# Compute partial densities
		arhoA  = self.y[0] * rho
		arhoWv = self.y[1] * rho
		arhoM  = self.y[2] * rho

		# Compute conservative variables for tracers (typically no source in 2D)
		arhoWt = arhoWv + self.tracer_frac * arhoM
		arhoC = self.tracer_frac * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM
		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute specific energy
		e_m = np.expand_dims(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p]), axis=-1)
		e_w = np.expand_dims(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, self.T), self.T)
			for _p in p]), axis=-1)
		rho = arhoA + arhoWv + arhoM
		# Compute volumetric energy
		# TODO: move implementation to middleware
		e = (arhoA * physics.Gas[0]["c_v"] * self.T + 
				arhoWv * e_w + 
				arhoM * (physics.Liquid["c_m"] * self.T + e_m)
				+ 0.5 * rho * (u*u+v*v)) # TODO: vpT analog needs correction for uv kinetc
		# Assemble conservative state vector		
		Uq[:, :, iarhoA] = arhoA
		Uq[:, :, iarhoWv] = arhoWv
		Uq[:, :, iarhoM] = arhoM
		Uq[:, :, irhou] = rho * u
		Uq[:, :, irhov] = rho * v
		Uq[:, :, ie] = e
		# Tracer quantities
		Uq[:, :, iarhoWt] = arhoWt
		Uq[:, :, iarhoC] = arhoC
		Uq[:, :, iarhoFm] = arhoFm

		return Uq # [ne, nq, ns]

'''
-------------------
Boundary conditions
-------------------
These classes inherit from either the BCWeakRiemann or BCWeakPrescribed
classes. See those parent classes for detailed comments of attributes
and methods. Information specific to the corresponding child classes can be
found below. These classes should correspond to the BCType enum members
above.
'''

# TODO: this thing
class LinearizedImpedance2D(BCWeakPrescribed):
	'''
	Impedance boundary condition linearized about the initial pressure
	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self):
		''' Pressure is initialized using the first encountered pressure. '''
		self.initialized = False

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		UqB = UqI.copy()
		''' Compute normal velocity. '''
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		# rhoI = UqI[:, :, physics.get_mass_slice()].sum(axis=2, keepdims=True)
		arhoVec = UqI[:,:,physics.get_mass_slice()]
		rhoI = atomics.rho(arhoVec)
		velI = UqI[:, :, physics.get_momentum_slice()]/rhoI
		# Normal velocity scalar
		veln = np.sum(velI * n_hat, axis=2, keepdims=True)
		# Normal velocity vector
		velvec_n = np.einsum("...i, ...i -> ...i", veln, n_hat)
		velvec_t = velI - velvec_n
		''' Inflow handling '''
		# if np.any(veln < 0.):
			# print("Incoming flow at outlet")
		''' Inflow handling '''
		# if np.any(velI < 0.): # TODO:
			# print("Incoming flow at outlet")
		''' Record first pressure encountered at boundary. '''
		if not self.initialized:
			self.p = physics.compute_variable("Pressure", UqI)
			self.initialized = True
		''' Compute interior pressure. '''
		# pI = physics.compute_variable("Pressure", UqI)
		# Linearized computation of outgoing stuff
		pGrid = physics.compute_variable("Pressure", UqI)
		ZGrid = physics.compute_variable("SoundSpeed", UqI) * atomics.rho(UqI[...,0:3])
		uGrid = veln
		TGrid = physics.compute_variable("Temperature", UqI)
		psiPlus = uGrid + (pGrid-self.p) / ZGrid
		uHat = 0.5 * psiPlus
		pHat = self.p + ZGrid * (0.5 * psiPlus)
		# Isentropic temp change
		Gamma = atomics.Gamma(UqI[...,0:3], physics)
		THat = TGrid * (pHat/pGrid)**((Gamma-1)/Gamma)

		if np.any(pHat < 0.):
			raise errors.NotPhysicalError
		''' Short-circuit function for sonic exit based on interior '''
		# cI = physics.compute_variable("SoundSpeed", UqI)		
		# cI = atomics.sound_speed(atomics.Gamma(arhoVec, physics),
		# 	pI, rhoI, gas_volfrac, physics)
		# if np.any(velI >= cI):
		# 	return UqB


		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to ingoing acoustic waves) of the interior
		solution. '''
		# Compute mass fractions of interior solution
		yI = atomics.massfrac(arhoVec)
		rho_target = atomics.mixture_density(yI, pHat, THat, physics)
		velvec_target = np.einsum("...i, ...i -> ...i", uHat, n_hat) \
			+ velvec_t
		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho_target * yI
		UqB[:,:,physics.get_momentum_slice()] = rho_target * uHat
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho_target * yI, physics) * THat \
			+ (rho_target * yI[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * rho_target * np.sum(velvec_target*velvec_target, axis=2, keepdims=True)
		''' Update adiabatically compressed/expanded tracer partial densities '''
		UqB[:,:,[physics.get_state_index("pDensityWt"),
		         physics.get_state_index("pDensityC")]] *= rho_target / rhoI

		''' Post-computation validity check '''
		if np.any(THat < 0.):
			raise errors.NotPhysicalError

		return UqB
