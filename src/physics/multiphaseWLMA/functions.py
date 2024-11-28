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
try:
	import physics.multiphaseWLMA.iapws95_light.mixtureWLMA as mixtureWLMA
except ModuleNotFoundError as e:
	# Module guard for users of main branch
	print("IAPWS95 water modules ignored. Error de-escalated to the following message: ")
	print(e.msg)

import errors
import general

from physics.base.data import (BCBase, FcnBase, BCWeakRiemann, BCWeakPrescribed,
				SourceBase, ConvNumFluxBase)
import physics.multiphasevpT.atomics as atomics

class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.s
	'''
	IsothermalAtmosphere = auto()
	OverOceanIsothermalAtmosphere = auto()
	UnderwaterMagmaConduit = auto()
	ThreeLayerModel = auto()
	CraterInitialVelocity = auto()

	# DebrisFlow = auto()
	# IsothermalAtmosphere1D = auto()


class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions.
	'''
	''' BCs specific to multiphaseWLMA (see also
	physics.multiphasevpT.functions). '''
	pass
	# SlipWall = auto()
	LinearizedImpedance2D = auto()
	PressureStableLinearizedInlet1D = auto()
	# MultiphasevpT1D1D = auto()
	# MultiphasevpT2D1D = auto()
	# PressureOutlet1D = auto()
	# MultiphasevpT2D2D = auto()
	# MultiphasevpT2D1DCylindrical = auto()
	# MultiphasevpT2D2DCylindrical = auto()
	# Not implemented (could use lumped magma chamber model for example)

class SourceType(Enum):
	''' SourceTypes specific to multiphaseWLMA (see also
	physics.multiphasevpT.functions). '''
	WaterMassSource = auto()
	WaterEntrainmentSource = auto()
	MagmaMassSource = auto()
	# FrictionVolFracVariableMu = auto()
	# FrictionVolFracConstMu = auto()
	GravitySource = auto()
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

	# Use 15e6 p_h0 for water (WLMA~11)
	def __init__(self,T:float=300., p_h0=35e6, p_surf=None, # # p_h0:float=15e6, #1e3*1300*9.8,
		h0:float=-800.0, hmax:float=2500.0, hmin:float=-1000.0, gravity:float=9.8,
		massFracWv:float=1.0-1e-7, massFracM:float=1e-7, tracer_frac:float=1e-7):
		''' Set atmosphere temperature, pressure, and location of pressure.
		Pressure distribution is computed as hydrostatic profile with p = p_h0
		at elevation h0.

		Uses p_surf if not None to specify the pressure at the surface.
		Else, uses p_h0 to specify the pressure at h0. 
		'''
		self.T = T
		self.p_h0 = p_h0
		self.p_surf = p_surf
		self.h0 = h0         # Surface at which p = p_h0, if p_surf==None
		self.hmax = hmax     # Max height to accommodate (across all domains)
		self.hmin = hmin     # Min height to accommodate (across all domains)
		if p_surf is not None:
			self.h0 = "ignored"
			self.p_h0 = "ignored"
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
		ndims = physics.NDIMS
		if ndims == 1:
			iarhoA, iarhoWv, iarhoM, irhou, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		elif ndims == 2:
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		else:
			raise ValueError(f"Unknown number of dimensions: {ndims}")

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

		''' Compute hydrostatic pressure as initial value problem (IVP) with global
		  limits and BC
  		[self.h0, self.hmax],
	  	[self.p_h0]
		  '''

		# TODO: replace air with water again
		# Water
		# self.y_underwater = np.array([self.tracer_frac,
			# 1.0-2.0*self.tracer_frac, self.tracer_frac])
		# Air
		self.y_underwater = np.array([1.0-2.0*self.tracer_frac,
			self.tracer_frac, self.tracer_frac])
		# Define evaluation points from input mesh y-coordinate
		eval_pts = np.unique(x[:,:,1:2])

		def mixture_spec_vol(p:float, T:float):
			# Volume and composition of water layer
			y = self.y_underwater
			# Aerated
			# return 1.2 \
			# 	+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
			# 	+ y[2] / (physics.Liquid["rho0"] 
			# 		* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))		
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))


		# Evaluate IVP solution
		if self.p_surf is not None:
			# Solve from surface downward
			soln = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol(p, self.T),
				[self.hmax, self.hmin],
				[self.p_surf],
				# t_eval=eval_pts,
				dense_output=True)
		else:
			# Solve from h0 upward
			soln = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol(p, self.T),
				[self.h0, self.hmax],
				[self.p_h0],
				# t_eval=eval_pts,
				dense_output=True)
		# Cache the pressure interpolant
		self.pressure_interpolant = soln.sol
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p = np.reshape(self.pressure_interpolant(x[...,-1].ravel()), x[...,-1].shape)
		# Constant-value extrapolation above hmax
		p = np.where(x[...,-1] <= self.hmax, p, self.p_surf)

		# Compute resultant density
		rho = np.reshape(
			1.0 / np.array([mixture_spec_vol(_p, self.T)
				for _p in p.ravel()]),
			p.shape)
		
		''' Use rho, p, T to fill in conserved variables '''

		# Compute conservative density variables
		arhoA = rho * self.y_underwater[0]
		arhoWv = rho * self.y_underwater[1]
		arhoM = rho * self.y_underwater[2]
		# Compute conservative variables for tracers (typically no source in 2D)
		arhoWt = arhoWv + self.tracer_frac * arhoM
		arhoC  = self.tracer_frac * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM

		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute specific energy for each phase
		e_a = physics.Gas[0]["c_v"] * self.T
		e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p.ravel()]), p.shape)
		e_m = physics.Liquid["c_m"] * self.T + e_m_mech
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, self.T), self.T)
			for _p in p.ravel()]), p.shape)
		rho = arhoA + arhoWv + arhoM
		# Compute volumetric energy
		# TODO: move implementation to middleware
		e_vol = (arhoA * e_a + arhoWv * e_w + arhoM * e_m
				 + 0.5 * rho * (u*u+v*v)) # TODO: vpT analog needs correction for uv kinetc
		# Assemble conservative state vector
		Uq[:, :, iarhoA]  = arhoA
		Uq[:, :, iarhoWv] = arhoWv
		Uq[:, :, iarhoM]  = arhoM
		Uq[:, :, irhou]   = rho * u
		if ndims == 2:
			Uq[:, :, irhov]   = rho * v
		Uq[:, :, ie]      = e_vol
		# Tracer quantities
		Uq[:, :, iarhoWt] = arhoWt
		Uq[:, :, iarhoC]  = arhoC
		Uq[:, :, iarhoFm] = arhoFm

		# Premix fill 2D HACK: hard-coded values instead of attaching premixing routine
		# select_yw = 0.95
		# U_premix = np.array([1.032568996254891e-04, 9.809405464421462e+02,
    #     5.162844981274459e+01, 0.000000000000000e+00,
    #     0.000000000000000e+00, 2.645152678731813e+08,
    #     9.809406496990458e+02, 1.032568996254891e-04,
    #     5.162844981274459e+01])
		# select_yw = 0.93
		# U_premix = np.array([1.045819302979261e-04, 9.726119517707126e+02,
		# 			7.320735120854820e+01, 0.000000000000000e+00,
		# 			0.000000000000000e+00, 3.283211682949423e+08,
		# 			9.726120563526429e+02, 1.045819302979261e-04,
		# 			7.320735120854820e+01])
		# select_yw = 0.94
		# U_premix = np.array([1.039151912411363e-04, 9.768027976666815e+02,
    #     6.234911474468186e+01, 0.000000000000000e+00,
    #     0.000000000000000e+00, 2.962148278533422e+08,
    #     9.768029015818727e+02, 1.039151912411363e-04,
    #     6.234911474468186e+01])
		# Uq = np.where((x[:,:,1:2] >= -300) & (x[:,:,1:2] <= -150) , U_premix, Uq)

		# select_yw = 0.90
		# U_premix = np.array([1.066344900380386e-04, 9.597104103423474e+02,
		# 			1.066344900380386e+02, 0.000000000000000e+00,
		# 			0.000000000000000e+00, 4.271607084743886e+08,
		# 			9.597105169768374e+02, 1.066344900380386e-04,
		# 			1.066344900380386e+02])

		# select_yw = 0.5
		# U_premix = np.array([1.444294213749651e-04, 7.221471068748257e+02,
    #     7.221471068748257e+02, 0.000000000000000e+00,
    #     0.000000000000000e+00, 2.247148483471262e+09,
    #     7.221472513042470e+02, 1.444294213749651e-04,
    #     7.221471068748257e+02])

		# select_yw = 0.75		
		U_premix = np.array([1.182373142717186e-04, 8.867798570378898e+02,
        2.955932856792966e+02, 0.000000000000000e+00,
        0.000000000000000e+00, 9.858863765404476e+08,
        8.867799752752039e+02, 1.182373142717186e-04,
        2.955932856792966e+02])

		if ndims==1:
			# Abridge for ndims == 1
			U_premix = U_premix[...,[0,1,2,3,5,6,7,8]]
		# Fill below 0 with premix
		# TODO: put premix back in
		# Uq = np.where(x[:,:,1:2] < 0 , U_premix, Uq)
		# Crater fill (must be < 0 to avoid filling the seabed)
		# Uq = np.where((x[:,:,1:2] >= -999999) & (x[:,:,1:2] < 0) , U_premix, Uq)
		# Dike fill
		# TODO: put the premixed section back in
		# Uq = np.where((x[:,:,1:2] > -500) & (x[:,:,1:2] < 0) , U_premix, Uq)
		# TODO: msh2, msh3 below
		# Uq = np.where((x[:,:,1:2] >= -50) & (x[:,:,1:2] < 0) , U_premix, Uq)

		''' Construct chamber fill '''
		self.y_chamber = np.array([self.tracer_frac,
			self.tracer_frac,
			1.0-2.0*self.tracer_frac])
		
		# Ellipsoidal parameters
		chamber_depth_center = -2500
		chamber_a = 2000 # Horizontal semiaxis
		chamber_b = 800 # Vertical semiaxis
		overpressure_chamber = 0.0 # TODO: check this is what we want
		T_chamber = 1300 # TODO: NOTE: Replaced

		chamber_top = chamber_depth_center + chamber_b
		chamber_bottom = chamber_depth_center - chamber_b
		p_waterhydrostatic_chamber_top = self.pressure_interpolant(0.0)[0] # TODO: replaced
		# p_waterhydrostatic_chamber_top = self.pressure_interpolant(chamber_top)[0] TODO: replaced

		p_chamber_top = p_waterhydrostatic_chamber_top + overpressure_chamber
		# Compute boolean array for whether a point x[ne, nb, ndim] is in chamber		
		is_in_chamber:np.array = (
			(x[...,0:1]*x[...,0:1])/(chamber_a * chamber_a)
			+ ((x[...,-1:]-chamber_depth_center)*(x[...,-1:]-chamber_depth_center))/(chamber_b * chamber_b) <= 1
		)
		# is_magmatic:np.array = is_in_chamber | (x[...,-1:] < 0)
		is_magmatic:np.array = is_in_chamber | (x[...,-1:] < -50)

		def mixture_spec_vol_chamber(p:float, T:float):
			y = self.y_chamber
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		# Solve downward the hydrostatic
		# TODO: changed for over_ocean
		# soln_chamber = scipy.integrate.solve_ivp(
		# 		lambda height, p:
		# 			-self.gravity / mixture_spec_vol_chamber(p, T_chamber),
		# 		[0.0, chamber_bottom],
		# 		# [chamber_top, chamber_bottom], # TODO: replaced by above
		# 		[p_chamber_top],
		# 		# t_eval=eval_pts,
		# 		dense_output=True)
		soln_chamber = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol_chamber(p, T_chamber),
				[self.hmax, self.hmin],
				# [chamber_top, chamber_bottom], # TODO: replaced by above
				[self.p_surf],
				# t_eval=eval_pts,
				dense_output=True)
		# Cache the pressure interpolant
		self.pressure_interpolant_chamber = soln_chamber.sol
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p_chamber = np.reshape(self.pressure_interpolant_chamber(x[...,-1].ravel()), x[...,-1].shape)
		# Constant-value extrapolation above hmax
		p_chamber = np.where(is_magmatic.squeeze(axis=-1), p_chamber, p_chamber_top)
		# Compute specific volume in chamber (not vectorized)
		v_chamber = np.reshape(np.array([mixture_spec_vol_chamber(_p, T_chamber)
			for _p in p_chamber.ravel()]), p_chamber.shape)
		rho_chamber = 1.0 / v_chamber
		# Compute chamber variables
		arhoA = rho_chamber * self.y_chamber[0]
		arhoWv = rho_chamber * self.y_chamber[1]
		arhoM = rho_chamber * self.y_chamber[2]
		# Compute conservative variables for tracers (typically no source in 2D)
		arhoWt = arhoWv + self.tracer_frac * arhoM
		arhoC  = self.tracer_frac * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM

		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute specific energy for each phase
		e_a = physics.Gas[0]["c_v"] * T_chamber
		e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p_chamber.ravel()]), p.shape)
		e_m = physics.Liquid["c_m"] * T_chamber + e_m_mech
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, T_chamber), T_chamber)
			for _p in p_chamber.ravel()]), p.shape)
		# Compute volumetric energy
		e_vol = (arhoA * e_a + arhoWv * e_w + arhoM * e_m
				 + 0.5 * rho * (u*u+v*v)) # TODO: vpT analog needs correction for uv kinetc
		# Assemble conservative state vector
		U_chamber = np.zeros_like(Uq)
		U_chamber[:, :, iarhoA]  = arhoA
		U_chamber[:, :, iarhoWv] = arhoWv
		U_chamber[:, :, iarhoM]  = arhoM
		U_chamber[:, :, irhou]   = rho_chamber * u
		if ndims == 2:
			U_chamber[:, :, irhov]   = rho_chamber * v
		U_chamber[:, :, ie]      = e_vol
		# Tracer quantities
		U_chamber[:, :, iarhoWt] = arhoWt
		U_chamber[:, :, iarhoC]  = arhoC
		U_chamber[:, :, iarhoFm] = arhoFm

		# mesh1, mesh2 setting
		# Uq = np.where((x[:,:,1:2] >= -99999) & (x[:,:,1:2] < -50) , U_chamber, Uq)
		# mesh3 setting
		# Uq = np.where((x[:,:,1:2] >= -99999) & (x[:,:,1:2] <= -500) , U_chamber, Uq)
		Uq = np.where(is_magmatic, U_chamber, Uq)

		# TODO: run#10 uses below
		# Uq = np.where((x[:,:,-1:] > chamber_top) & (x[:,:,-1:] < 0) , U_premix, Uq)

		return Uq # [ne, nq, ns]

class OverOceanIsothermalAtmosphere(FcnBase):
	'''
	Isothermal air atmosphere as an initial condition, over ocean layer.
	'''

	def __init__(self, T:float=300., psurf=1e5,
		hsurf:float=0.0, hmin:float=-50, hmax:float=20000.0, gravity:float=9.8,
		tracer_frac:float=1e-7, include_water_layer:bool=True,
		tracer_frac_w:float=1e-3, tracer_frac_m:float=1e-7):

		self.T = T
		self.psurf = psurf
		self.hsurf = hsurf
		self.hmin = hmin
		self.hmax = hmax
		self.gravity = gravity
		self.tracer_frac = tracer_frac # numerical fraction for essentially inert fields
		# Allocate pressure interpolant (filled when self.get_state is called
		# because physics object is required)
		self.include_water_layer = include_water_layer
		# Individual tracer fractions for water and magma separately, for atmosphere
		self.tracer_frac_w = tracer_frac_w
		self.tracer_frac_m = tracer_frac_m
		
	def get_state(self, physics, x, t):
		''' Computes the pressure in an isothermal atmosphere for air. Trace
		amounts of magma phase are added and the pressure profile is iteratively
		corrected. '''

		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		ndims = physics.NDIMS
		if ndims == 1:
			iarhoA, iarhoWv, iarhoM, irhou, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		elif ndims == 2:
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		else:
			raise ValueError(f"Unknown number of dimensions: {ndims}")

		# Mixture mass fraction vector
		self.y_overwater = np.array([
			1.0-(self.tracer_frac_w + self.tracer_frac_m),
			self.tracer_frac_w,
			self.tracer_frac_m])
		self.y_underwater = np.array([
			self.tracer_frac,
			1.0-2.0*self.tracer_frac,
			self.tracer_frac]) if self.include_water_layer else self.y_overwater

		# self.y_underwater = self.y_overwater ############################################################### DRY

		def mixture_spec_vol(p:float, T:float, y:list) -> float:
			''' Evaluate v(p,T,y), non-vectorized. '''
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		def compute_isothermal_layer(z:np.array, T:float, y:list,
															   h0:float, h1:float, p0:float) -> np.array:
			''' Compute U for an isothermal layer, evaluated on points in z.ravel(). '''
			soln = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol(p, T, y),
				[h0, h1],
				[p0],
				dense_output=True)
			# Evaluate p using dense_output of solve_ivp (with shape magic)
			p = np.reshape(soln.sol(z.ravel()), z.ravel().shape)
			# Compute resultant density
			rho = np.reshape(
				1.0 / np.array([mixture_spec_vol(_p, T, y)
					for _p in p.ravel()]),
				p.shape)
			# Use rho, p, T to fill in conserved variables
			arhoA = rho * y[0]
			arhoWv = rho * y[1]
			arhoM = rho * y[2]
			# Compute conservative variables for tracers (typically no source in 2D)
			arhoWt = arhoWv + self.tracer_frac * arhoM
			arhoC  = self.tracer_frac * arhoM
			arhoFm = (1.0 - self.tracer_frac) * arhoM
			# Zero velocity
			u = np.zeros_like(p)
			v = np.zeros_like(p)
			# Compute specific energy for each phase
			e_a = physics.Gas[0]["c_v"] * T
			e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
				physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
				for _p in p.ravel()]), p.shape)
			e_m = physics.Liquid["c_m"] * T + e_m_mech
			e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
				mixtureWLMA.float_mix_functions.rho_l_pt(_p, T), T)
				for _p in p.ravel()]), p.shape)
			rho = arhoA + arhoWv + arhoM
			# Compute volumetric energy
			e_vol = (arhoA * e_a + arhoWv * e_w + arhoM * e_m
					+ 0.5 * rho * (u*u+v*v))
			# Assemble conservative state vector
			Uq = np.zeros((z.ravel().size, physics.NUM_STATE_VARS))
			Uq[:, iarhoA]  = arhoA
			Uq[:, iarhoWv] = arhoWv
			Uq[:, iarhoM]  = arhoM
			Uq[:, irhou]   = rho * u
			if physics.NDIMS == 2:
				Uq[:, irhov]   = rho * v
			Uq[:, ie]      = e_vol
			# Tracer quantities
			Uq[:, iarhoWt] = arhoWt
			Uq[:, iarhoC]  = arhoC
			Uq[:, iarhoFm] = arhoFm
			return Uq
		
		# Air over water
		ind_ravel_over_water = np.where(x[...,-1].ravel() >= 0)
		ind_ravel_under_water = np.where(x[...,-1].ravel() < 0)
		# Flatten Uq vector to (ndof, ns)
		Uq_flat:np.ndarray = np.reshape(Uq, (-1, physics.NUM_STATE_VARS,))

		Uq_flat[ind_ravel_over_water, :] = compute_isothermal_layer(
			x[...,-1].ravel()[ind_ravel_over_water],
			self.T,
			self.y_overwater,
			self.hsurf,
			self.hmax, 
			self.psurf)
		if ind_ravel_under_water[0].size > 0:
			Uq_flat[ind_ravel_under_water, :] = compute_isothermal_layer(
				x[...,-1].ravel()[ind_ravel_under_water],
				self.T,
				self.y_underwater,
				self.hsurf,
				self.hmin, 
				self.psurf)
		Uq = np.reshape(Uq_flat, Uq.shape)

		return Uq # [ne, nq, ns]

class UnderwaterMagmaConduit(FcnBase):
	'''
	Compute overpressurized magma state under water layer.

	hwm: water-magma interface height
	'''

	def __init__(self, T_magma:float=1273.15, T_water:float=300,
							 psurf:float=1e5, hsurf:float=0.0,
							 hwm:float=-300, hmin:float=-2050,
							 delta_p:float=10e6, gravity:float=9.8,
							 tracer_frac:float=1e-7, yC:float=1e-7, yWt:float=0.05,
							 solubility_k:float=5e-6, solubility_n:float=0.5):

		self.T_magma = T_magma
		self.T_water = T_water
		self.psurf = psurf
		self.hsurf = hsurf
		self.hwm = hwm # water-magma interface height
		self.hmin = hmin
		self.delta_p = delta_p
		self.gravity = gravity
		self.tracer_frac = tracer_frac # numerical fraction for essentially inert fields
		self.yC = yC # numerical fraction for essentially inert fields
		self.yWt = yWt
		self.solubility_k = solubility_k
		self.solubility_n = solubility_n
		
	def get_state(self, physics, x, t):
		''' Computes the pressure in an isothermal atmosphere for air. Trace
		amounts of magma phase are added and the pressure profile is iteratively
		corrected. '''

		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		ndims = physics.NDIMS
		if ndims == 1:
			iarhoA, iarhoWv, iarhoM, irhou, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		elif ndims == 2:
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		else:
			raise ValueError(f"Unknown number of dimensions: {ndims}")

		def mixture_spec_vol(p:float, T:float, y:list) -> float:
			''' Evaluate v(p,T,y), non-vectorized. '''
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		''' Compute additional confining pressure from water layer '''
		self.y_underwater = np.array([self.tracer_frac,
			1.0-2.0*self.tracer_frac, self.tracer_frac])
		soln = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol(p, self.T_water, self.y_underwater),
				[0.0, -100+self.hwm],
				# [self.hsurf, self.hwm],
				[self.psurf],
				# [self.psurf],
				t_eval=[-100, -100+self.hwm], # Manual input for height 2D surface
				# t_eval=[self.hwm],
				)
		# Compute pressure at top of 1D domain (water)
		p_wtop = soln.y[0,0]
		# Compute pressure at water-magma interface
		p_mtop = soln.y[0,1] + self.delta_p

		def compute_isothermal_layer(z:np.array, T:float, y:list,
															   h0:float, h1:float, p0:float) -> np.array:
			''' Compute U for an isothermal layer, evaluated on points in z.ravel(). '''
			soln = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol(p, T, y),
				[h0, h1],
				[p0],
				dense_output=True)
			# Evaluate p using dense_output of solve_ivp (with shape magic)
			p = np.reshape(soln.sol(z.ravel()), z.ravel().shape)
			# Compute resultant density
			rho = np.reshape(
				1.0 / np.array([mixture_spec_vol(_p, T, y)
					for _p in p.ravel()]),
				p.shape)
			# Use rho, p, T to fill in conserved variables
			arhoA = rho * y[0]
			arhoWv = rho * y[1]
			arhoM = rho * y[2]
			# Compute conservative variables for tracers (typically no source in 2D)
			arhoWt = arhoWv + self.tracer_frac * arhoM
			arhoC  = self.tracer_frac * arhoM
			arhoFm = (1.0 - self.tracer_frac) * arhoM
			# Zero velocity
			u = np.zeros_like(p)
			v = np.zeros_like(p)
			# Compute specific energy for each phase
			e_a = physics.Gas[0]["c_v"] * T
			e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
				physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
				for _p in p.ravel()]), p.shape)
			e_m = physics.Liquid["c_m"] * T + e_m_mech
			e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
				mixtureWLMA.float_mix_functions.rho_l_pt(_p, T), T)
				for _p in p.ravel()]), p.shape)
			rho = arhoA + arhoWv + arhoM
			# Compute volumetric energy
			e_vol = (arhoA * e_a + arhoWv * e_w + arhoM * e_m
					+ 0.5 * rho * (u*u+v*v))
			# Assemble conservative state vector
			Uq = np.zeros((z.ravel().size, physics.NUM_STATE_VARS))
			Uq[:, iarhoA]  = arhoA
			Uq[:, iarhoWv] = arhoWv
			Uq[:, iarhoM]  = arhoM
			Uq[:, irhou]   = rho * u
			if physics.NDIMS == 2:
				Uq[:, irhov]   = rho * v
			Uq[:, ie]      = e_vol
			# Tracer quantities
			Uq[:, iarhoWt] = arhoWt
			Uq[:, iarhoC]  = arhoC
			Uq[:, iarhoFm] = arhoFm

			return Uq

		def magma_spec_vol(p:float) -> float:
			''' Evaluate magma specific volume with equilibrium dissolution. '''
			# Compute dissolved (yWd) and exsolved (yWv) mass fractions
			yL = 1.0 - (self.yWt + self.yC)
			yWd_max =	yL * self.solubility_k * p ** self.solubility_n
			yWd = np.clip(yWd_max, self.tracer_frac, self.yWt - self.tracer_frac)
			yWv = self.yWt - yWd
			# Compute other mass fractions
			yA = self.tracer_frac
			yM = 1.0 - (yA + yWv)

			return mixture_spec_vol(p, self.T_magma, [yA, yWv, yM])

		soln = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / magma_spec_vol(p),
			[self.hwm, self.hmin],
			[p_mtop],
			dense_output=True)
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p = np.reshape(soln.sol(x[...,-1].ravel()), x[...,-1].ravel().shape)
		# Compute resultant density
		rho = np.reshape(
			1.0 / np.array([magma_spec_vol(_p)
				for _p in p.ravel()]),
			p.shape)
		# Use rho, p, T to fill in conserved variables
		yL = 1.0 - (self.yWt + self.yC)
		yWd_max =	yL * self.solubility_k * p ** self.solubility_n
		yWd = np.clip(yWd_max, self.tracer_frac, self.yWt - self.tracer_frac)
		yWv = self.yWt - yWd
		# Compute other mass fractions, partial densities
		yA = self.tracer_frac
		yM = 1.0 - (yA + yWv)
		arhoA  = rho * yA
		arhoWv = rho * yWv
		arhoM  = rho * yM
		arhoWt = rho * self.yWt
		arhoC  = rho * self.yC
		arhoFm = self.tracer_frac * arhoM
		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute specific energy for each phase
		e_a = physics.Gas[0]["c_v"] * self.T_magma
		e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p.ravel()]), p.shape)
		e_m = physics.Liquid["c_m"] * self.T_magma + e_m_mech
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, self.T_magma), self.T_magma)
			for _p in p.ravel()]), p.shape)
		rho = arhoA + arhoWv + arhoM
		# Compute volumetric energy
		e_vol = (arhoA * e_a + arhoWv * e_w + arhoM * e_m
				+ 0.5 * rho * (u*u+v*v))
		# Assemble conservative state vector
		Uq[..., iarhoA] = np.reshape(arhoA, Uq.shape[0:2])
		Uq[..., iarhoWv] = np.reshape(arhoWv, Uq.shape[0:2])
		Uq[..., iarhoM] = np.reshape(arhoM, Uq.shape[0:2])
		Uq[..., irhou] = np.reshape(rho * u, Uq.shape[0:2])
		if physics.NDIMS == 2:
			Uq[..., irhov] = np.reshape(rho * v, Uq.shape[0:2])
		Uq[..., ie] = np.reshape(e_vol, Uq.shape[0:2])
		Uq[..., iarhoWt] = np.reshape(arhoWt, Uq.shape[0:2])
		Uq[..., iarhoC] = np.reshape(arhoC, Uq.shape[0:2])
		Uq[..., iarhoFm] = np.reshape(arhoFm, Uq.shape[0:2])

		# REplace top layer water
		# p_wtop
		# Uq_flat:np.ndarray = np.reshape(Uq, (-1, physics.NUM_STATE_VARS,))

		# Uq_water = np.empty_like(Uq)
		Uq_water = np.reshape(compute_isothermal_layer(
			x.ravel(),
			self.T_water,
			self.y_underwater,
			-50,
			self.hmin, # Overextend
			p_wtop), Uq.shape)
		
		# Modify dissolved water in magma to saturation using one-step correction
		# Extract p (redundant with compute_isothermal_layer but cheap)
		soln = scipy.integrate.solve_ivp(
						lambda height, p:
							-self.gravity / mixture_spec_vol(p, self.T_water, self.y_underwater),
						[-50, self.hmin],
						[p_wtop],
						dense_output=True)
		_p = np.reshape(soln.sol(x.ravel()), Uq_water[...,0:1].shape)
		_rho = Uq_water[...,0:3].sum(axis=-1, keepdims=True)
		_spec_Uq_water = Uq_water / _rho
		_yL = 1.0 - (_spec_Uq_water[...,5:6] + _spec_Uq_water[...,6:7])
		yWd_corr = _yL * self.solubility_k * _p ** self.solubility_n
		# Apply correction to total water content
		Uq_water[...,5:6] = (_spec_Uq_water[...,1:2] + yWd_corr) * _rho

		# Combine 1D water layer and 1D magma
		Uq = np.reshape(np.where(x[...,-1].ravel()[...,np.newaxis] <= self.hwm,
								np.reshape(Uq, (-1, physics.NUM_STATE_VARS)),
								np.reshape(Uq_water, (-1, physics.NUM_STATE_VARS))), Uq.shape)

		return Uq # [ne, nq, ns]


class CraterInitialVelocity(FcnBase):
	'''
	Initial upward velocity in an instantaneously excavated crater
	'''

	def __init__(self, p_inf:float=1.1e6, delta_p:float=0.0,
							 T_magma:float=1300, T_inf:float=300,
							 z_virtual:float=-10, tracer_frac:float=1e-5,
							 launch_speed:float=0.0, crater_depth:float=5,
							 solubility_k:float=5e-6, solubility_n:float=0.5):

		self.p_inf = p_inf
		self.delta_p = delta_p
		self.T_magma = T_magma
		self.T_inf = T_inf
		self.z_virtual = z_virtual # Virtual apex of conical crater # -10
		self.tracer_frac = tracer_frac
		self.launch_speed = launch_speed
		self.crater_depth = crater_depth

		self.solubility_k = solubility_k
		self.solubility_n = solubility_n

		''' Compute additional confining pressure from water layer '''
		self.y_magmatic = np.array([tracer_frac,
															  tracer_frac,
																1.0 - 2*tracer_frac])
		self.y_water = np.array([tracer_frac,
													 	 1.0 - 2*tracer_frac,
													 	 tracer_frac])
		
	def get_state(self, physics, x, t):
		''' Computes the pressure in an isothermal atmosphere for air. Trace
		amounts of magma phase are added and the pressure profile is iteratively
		corrected. '''

		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		ndims = physics.NDIMS
		if ndims == 1:
			iarhoA, iarhoWv, iarhoM, irhou, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		elif ndims == 2:
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		else:
			raise ValueError(f"Unknown number of dimensions: {ndims}")

		def mixture_spec_vol(p:float, T:float, y:list) -> float:
			''' Evaluate v(p,T,y), non-vectorized. '''
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		v_ocean = mixture_spec_vol(self.p_inf, self.T_inf, self.y_water)
		v_magma = mixture_spec_vol(self.p_inf + self.delta_p,
														 self.T_magma,
														 self.y_magmatic)
		rho_ocean = 1.0 / v_ocean
		rho_magma = 1.0 / v_magma
		

		# Compute velocity ray path
		if ndims == 1:
			raise NotImplementedError("Conical crater ray path not calculated for 1D.")
		# Unit vector from virtual apex to location
		direction = x.copy()
		direction[...,1] -= self.z_virtual
		direction /= np.linalg.norm(direction, axis=-1, keepdims=True)

		# TODO: Add exsolved water
		# def magma_spec_vol(p:float) -> float:
		# 	''' Evaluate magma specific volume with equilibrium dissolution. '''
		# 	# Compute dissolved (yWd) and exsolved (yWv) mass fractions
		# 	yL = 1.0 - (self.yWt + self.yC)
		# 	yWd_max =	yL * self.solubility_k * p ** self.solubility_n
		# 	yWd = np.clip(yWd_max, self.tracer_frac, self.yWt - self.tracer_frac)
		# 	yWv = self.yWt - yWd
		# 	# Compute other mass fractions
		# 	yA = self.tracer_frac
		# 	yM = 1.0 - (yA + yWv)
		# 	return mixture_spec_vol(p, self.T_magma, [yA, yWv, yM])
		# Use rho, p, T to fill in conserved variables
		# yL = 1.0 - (self.yWt + self.yC)
		# yWd_max =	yL * self.solubility_k * p ** self.solubility_n
		# yWd = np.clip(yWd_max, self.tracer_frac, self.yWt - self.tracer_frac)
		# yWv = self.yWt - yWd
		# # Compute other mass fractions, partial densities
		# yA = self.tracer_frac
		# yM = 1.0 - (yA + yWv)
		# arhoA  = rho * yA
		# arhoWv = rho * yWv
		# arhoM  = rho * yM
		# arhoWt = rho * self.yWt
		# arhoC  = rho * self.yC
		# arhoFm = self.tracer_frac * arhoM

		# Compute specific energy for each phase
		def e_mix(p, T, y):
			e_a = physics.Gas[0]["c_v"] * T
			e_m_mech = mixtureWLMA.float_mix_functions.magma_mech_energy(self.p_inf,
				physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			e_m = physics.Liquid["c_m"] * T + e_m_mech
			e_wv = mixtureWLMA.float_mix_functions.u(
				mixtureWLMA.float_mix_functions.rho_l_pt(p, T), T)
			return	y[0] * e_a + y[1] * e_wv + y[2] * e_m

		# Mixture energy of water
		e_water = e_mix(self.p_inf, self.T_inf, self.y_water)
		# Assemble conservative state vector for ocean
		Uq[..., iarhoA] = rho_ocean * self.y_water[0]
		Uq[..., iarhoWv] = rho_ocean * self.y_water[1]
		Uq[..., iarhoM] = rho_ocean * self.y_water[2]
		Uq[..., irhou] = 0.0
		if physics.NDIMS == 2:
			Uq[..., irhov] = 0.0
		Uq[..., ie] = rho_ocean * e_water
		Uq[..., iarhoWt] = rho_ocean * self.y_water[1]
		Uq[..., iarhoC] = 0.0
		Uq[..., iarhoFm] = rho_ocean * self.y_water[2]

		Uq_magma = Uq.copy()

		# Mixture energy of magma
		e_magmatic = e_mix(self.p_inf + self.delta_p, self.T_magma, self.y_magmatic)
		# Assemble conservative state vector for static magma
		Uq_magma[..., iarhoA] = rho_ocean * self.y_magmatic[0]
		Uq_magma[..., iarhoWv] = rho_ocean * self.y_magmatic[1]
		Uq_magma[..., iarhoM] = rho_ocean * self.y_magmatic[2]
		Uq_magma[..., iarhoWt] = rho_ocean * self.y_magmatic[1]
		Uq_magma[..., iarhoC] = 0.0
		Uq_magma[..., iarhoFm] = rho_ocean * self.y_magmatic[2]
		if physics.NDIMS == 1:
			Uq_magma[..., irhou] = rho_magma * 0.0
		elif physics.NDIMS == 2:
			Uq_magma[..., irhou] = rho_magma * 0.0
			Uq_magma[..., irhov] = rho_magma * 0.0
		Uq_magma[..., ie] = rho_magma * e_magmatic

		Uq_plug = Uq_magma.copy()
		# Set launch velocity
		u = self.launch_speed * direction[...,0]
		v = self.launch_speed * direction[...,1]
		if physics.NDIMS == 1:
			Uq_plug[..., irhou] = rho_magma * v
		elif physics.NDIMS == 2:
			Uq_plug[..., irhou] = rho_magma * u
			Uq_plug[..., irhov] = rho_magma * v
		Uq_plug[..., ie] += 0.5 * rho_magma * (u * u + v * v)
		
		# Combine piecewise U according to last dimension
		Uq = np.where(x[...,-1:] >= 0.0,
			Uq,
			np.where(x[...,-1:] >= -self.crater_depth,
				Uq_plug,
				Uq_magma)
		)

		return Uq # [ne, nq, ns]


class ThreeLayerModel(FcnBase):
	'''
	Compute overpressurized magma state under water layer.

	hwm: water-magma interface height
	'''

	def __init__(self,
							 z_ocean_surface=50,
							 delta_p=10e6,
							 T_water=300,
							 T_infiltrate=300,
							 T_magma=1000,
							 porosity=0.5,
							 z_vent=-100,
							 z_premix=-300,
							 psurf=1e5,
							 z_min:float=-2050,
							 gravity:float=9.8,
							 tracer_frac:float=1e-7,
							 yC:float=1e-7,
							 yWt:float=0.05,
							 solubility_k:float=5e-6,
							 solubility_n:float=0.5):
		self.z_ocean_surface = z_ocean_surface
		self.delta_p = delta_p
		self.T_water = T_water
		self.T_infiltrate = T_infiltrate
		self.T_magma = T_magma
		self.porosity = porosity
		self.z_vent = z_vent
		self.z_premix= z_premix
		self.psurf = psurf
		self.z_min = z_min
		self.gravity = gravity
		self.tracer_frac = tracer_frac     # numerical fraction for essentially inert fields
		self.yC = yC
		self.yWt = yWt
		self.solubility_k = solubility_k
		self.solubility_n = solubility_n

	def get_state(self, physics, x, t):
		''' Computes the pressure in an isothermal atmosphere for air. Trace
		amounts of magma phase are added and the pressure profile is iteratively
		corrected. '''

		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		ndims = physics.NDIMS
		if ndims == 1:
			iarhoA, iarhoWv, iarhoM, irhou, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		elif ndims == 2:
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
		else:
			raise ValueError(f"Unknown number of dimensions: {ndims}")

		def mixture_spec_vol(p:float, T:float, y:list) -> float:
			''' Evaluate v(p,T,y), non-vectorized. '''
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		def mixture_spec_vol_eq(p:float, T:float, yWt:float) -> float:
			''' Evaluate v(p,T,y), non-vectorized. For magma with equilibrium solubility. '''
			yWd = self.solubility_k * p ** self.solubility_n
			return self.tracer_frac * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))
		
		''' Compute additional confining pressure from water layer '''
		# self.y_underwater = np.array([1.0-2e-4, # air replacement ##############################################
		# 	1e-4, 1e-4])
		self.y_underwater = np.array([1e-8,
			1.0-2.0*self.tracer_frac, 1e-8])
		
		# Pure hydrostatic solution
		hydrostatic_soln = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol(p, self.T_water, self.y_underwater),
				[self.z_ocean_surface, self.z_min],
				[self.psurf],
				t_eval=[0.0, self.z_vent, self.z_premix],
				dense_output=True, # Manual input for height 2D surface
				)
		# Compute pressure at top of 1D domain (water)
		p_wtop = hydrostatic_soln.y[0,0]
		# Compute pressure at vent
		p_vent = hydrostatic_soln.y[0,1]

		def compute_isothermal_water_layer(z:np.array, T:float, y:list,
															   h0:float, h1:float, p0:float) -> np.array:
			''' Compute U for an isothermal layer, evaluated on points in z.ravel(). '''
			soln = scipy.integrate.solve_ivp(
				lambda z, p:
					-self.gravity / mixture_spec_vol(p, T, self.y_underwater),
				[h0, h1],
				[p0],
				dense_output=True)
			# Evaluate p using dense_output of solve_ivp (with shape magic)
			p = np.reshape(soln.sol(z.ravel()), z.ravel().shape)
			# Compute resultant density
			rho = np.reshape(
				1.0 / np.array([mixture_spec_vol(_p, T, y)
					for _p in p.ravel()]),
				p.shape)
			# Use rho, p, T to fill in conserved variables
			arhoA = rho * y[0]
			arhoWv = rho * y[1]
			arhoM = rho * y[2]
			# Compute conservative variables for tracers (typically no source in 2D)
			arhoWt = arhoWv + self.tracer_frac * arhoM
			arhoC  = self.tracer_frac * arhoM
			arhoFm = (1.0 - self.tracer_frac) * arhoM
			# Zero velocity
			u = np.zeros_like(p)
			v = np.zeros_like(p)
			# Compute specific energy for each phase
			e_a = physics.Gas[0]["c_v"] * T
			e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
				physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
				for _p in p.ravel()]), p.shape)
			e_m = physics.Liquid["c_m"] * T + e_m_mech
			e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
				mixtureWLMA.float_mix_functions.rho_l_pt(_p, T), T)
				for _p in p.ravel()]), p.shape)
			rho = arhoA + arhoWv + arhoM
			# Compute volumetric energy
			e_vol = (arhoA * e_a + arhoWv * e_w + arhoM * e_m
					+ 0.5 * rho * (u*u+v*v))
			# Assemble conservative state vector
			Uq = np.zeros((z.ravel().size, physics.NUM_STATE_VARS))
			Uq[:, iarhoA]  = arhoA
			Uq[:, iarhoWv] = arhoWv
			Uq[:, iarhoM]  = arhoM
			Uq[:, irhou]   = rho * u
			if physics.NDIMS == 2:
				Uq[:, irhov]   = rho * v
			Uq[:, ie]      = e_vol
			# Tracer quantities
			Uq[:, iarhoWt] = arhoWt
			Uq[:, iarhoC]  = arhoC
			Uq[:, iarhoFm] = arhoFm

			return Uq

		def magma_spec_vol(p:float) -> float:
			''' Evaluate magma specific volume with equilibrium dissolution. '''
			# Compute dissolved (yWd) and exsolved (yWv) mass fractions
			yL = 1.0 - (self.yWt + self.yC)
			yWd_max =	yL * self.solubility_k * p ** self.solubility_n
			yWd = np.clip(yWd_max, self.tracer_frac, self.yWt - self.tracer_frac)
			yWv = self.yWt - yWd
			# Compute other mass fractions
			yA = self.tracer_frac
			yM = 1.0 - (yA + yWv)

			return mixture_spec_vol(p, self.T_magma, [yA, yWv, yM])

		''' Build hydrostatic middle premix layer '''
		hydrostatic_soln = scipy.integrate.solve_ivp(
				lambda height, p:
					-self.gravity / mixture_spec_vol(p, self.T_water, self.y_underwater),
				[self.z_vent, self.z_min],
				[p_vent + self.delta_p],
				dense_output=True, # Manual input for height 2D surface
				)
		p_hydrostatic = hydrostatic_soln.sol(x.ravel()).ravel()
		# Compute rho_l as a function of p (applies when p > p_sat(T))
		# rhol_unfiltered = mixtureWLMA.rhoc * np.array([
		# 	mixtureWLMA.float_phi_functions.rhol_pT(p, self.T_infiltrate) for p in p_hydrostatic])
		psat = mixtureWLMA.float_phi_functions.prho_sat(self.T_infiltrate)["psat"]
		rhol_T = mixtureWLMA.float_phi_functions.prho_sat(self.T_infiltrate)["rho_satl"]
		rhov_T = mixtureWLMA.float_phi_functions.prho_sat(self.T_infiltrate)["rho_satv"]
		
		if self.T_infiltrate <= mixtureWLMA.Tc:
			rho_range_l = np.linspace(rhol_T, 1.3*rhol_T, 1000)
			rho_to_p = [mixtureWLMA.float_phi_functions.p(_rho, self.T_infiltrate)
									for _rho in rho_range_l]
			rho_l = np.interp(p_hydrostatic, rho_to_p, rho_range_l)

			# Compute rho_v as a function of p (applies when p < p_sat(T))
			rho_range_v = np.linspace(1e-1, rhov_T, 1000)
			rho_to_p = [mixtureWLMA.float_phi_functions.p(_rho, self.T_infiltrate)
									for _rho in rho_range_v]
			rho_v = np.interp(p_hydrostatic, rho_to_p, rho_range_v)
			rho_w_pore = np.where(p_hydrostatic < psat, rho_v, rho_l)
		else:
			rho_range = np.linspace(0.01, 1.2e3, 1000)
			rho_to_p = [mixtureWLMA.float_phi_functions.p(_rho, self.T_infiltrate)
							  for _rho in rho_range]
			rho_w_pore = np.interp(p_hydrostatic, rho_to_p, rho_range)


		# Compute density of condensed magma phase
		rho_m = physics.Liquid["rho0"] * (
			1 + (p_hydrostatic - physics.Liquid["p0"]) / physics.Liquid["K"])
		# Compute partial densities
		arhoWv = self.porosity * rho_w_pore
		arhoM = (1 - self.porosity) * rho_m

		# Compute conservative variables for tracers (typically no source in 2D)
		arhoA = self.tracer_frac * arhoWv
		arhoWt = arhoWv + self.tracer_frac * arhoM
		arhoC  = (1.0 - self.tracer_frac) * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM
		# Zero velocity
		u = np.zeros_like(p_hydrostatic)
		v = np.zeros_like(p_hydrostatic)
		# Compute specific energy for each phase
		e_a = physics.Gas[0]["c_v"] * self.T_infiltrate
		e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p_hydrostatic.ravel()]), p_hydrostatic.shape)
		e_m = physics.Liquid["c_m"] * self.T_infiltrate + e_m_mech
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(rho, self.T_infiltrate)
			for rho in rho_w_pore.ravel()]), p_hydrostatic.shape)
		rho = arhoA + arhoWv + arhoM
		# Compute volumetric energy
		e_vol = (arhoA * e_a + arhoWv * e_w + arhoM * e_m
				+ 0.5 * rho * (u*u+v*v))
		# Assemble conservative state vector
		Uq_premix = Uq.copy()
		Uq_premix[:, :, iarhoA]  = np.reshape(arhoA, Uq_premix.shape[0:2])
		Uq_premix[:, :, iarhoWv] = np.reshape(arhoWv, Uq_premix.shape[0:2])
		Uq_premix[:, :, iarhoM]  = np.reshape(arhoM, Uq_premix.shape[0:2])
		Uq_premix[:, :, irhou]   = np.reshape(rho * u, Uq_premix.shape[0:2])
		if physics.NDIMS == 2:
			Uq_premix[:, :, irhov]   = np.reshape(rho * v, Uq_premix.shape[0:2])
		Uq_premix[:, :, ie]      = np.reshape(e_vol, Uq_premix.shape[0:2])
		# Tracer quantities
		Uq_premix[:, :, iarhoWt] = np.reshape(arhoWt, Uq_premix.shape[0:2])
		Uq_premix[:, :, iarhoC]  = np.reshape(arhoC, Uq_premix.shape[0:2])
		Uq_premix[:, :, iarhoFm] = np.reshape(arhoFm, Uq_premix.shape[0:2])

		''' Build water layer '''
		Uq_water = np.reshape(compute_isothermal_water_layer(
			x.ravel(),
			self.T_water,
			self.y_underwater,
			0,
			self.z_min, # Overextend, and clip after
			p_wtop), Uq.shape)
		
		
		''' Build magma underlayer '''
		
		# Modify dissolved water in magma to saturation using one-step correction
		# Extract p (redundant with compute_isothermal_layer but cheap)
		soln = scipy.integrate.solve_ivp(
						lambda height, p:
							-self.gravity / magma_spec_vol(p),
						[self.z_premix, self.z_min],
						hydrostatic_soln.sol(self.z_premix),
						dense_output=True)
		p = np.reshape(soln.sol(x.ravel()), Uq_water[...,0:1].shape)

		# Compute dissolved (yWd) and exsolved (yWv) mass fractions
		yL = 1.0 - (self.yWt + self.yC)
		yWd_max =	yL * self.solubility_k * p ** self.solubility_n
		yWd = np.clip(yWd_max, self.tracer_frac, self.yWt - self.tracer_frac)
		yWv = self.yWt - yWd
		# Compute other mass fractions
		yA = self.tracer_frac * np.ones_like(yWv)
		yM = 1.0 - (yA + yWv)
		yF = np.zeros_like(yM)

		rho = np.reshape(
				1.0 / np.array([mixture_spec_vol(_p, self.T_magma, _y)
					for (_p, _y) in zip(
						p.ravel(),
						np.stack([yA.ravel(), yWv.ravel(), yM.ravel()], axis=-1))]),
				p.shape)
		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute specific energy for each phase
		e_a = physics.Gas[0]["c_v"] * self.T_magma
		e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p.ravel()]), p.shape)
		e_m = physics.Liquid["c_m"] * self.T_magma + e_m_mech
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, self.T_magma), self.T_magma)
			for _p in p.ravel()]), p.shape)
		# Compute volumetric energy
		e_vol = rho * (yA * e_a + yWv * e_w + yM * e_m
				+ 0.5 * (u*u+v*v))

		Uq_magma = Uq_water.copy()
		Uq_magma[:, :, iarhoA]  = (rho * yA).squeeze(axis=2)
		Uq_magma[:, :, iarhoWv] = (rho * yWv).squeeze(axis=2)
		Uq_magma[:, :, iarhoM]  = (rho * yM).squeeze(axis=2)
		Uq_magma[:, :, irhou]   = np.reshape(rho * u, Uq_magma.shape[0:2])
		if physics.NDIMS == 2:
			Uq_magma[:, :, irhov]   = np.reshape(rho * v, Uq_magma.shape[0:2])
		Uq_magma[:, :, ie]      = np.reshape(e_vol, Uq_magma.shape[0:2])
		# Tracer quantities
		Uq_magma[:, :, iarhoWt] = (rho * self.yWt).squeeze(axis=2)
		Uq_magma[:, :, iarhoC]  = (rho * self.yC).squeeze(axis=2)
		Uq_magma[:, :, iarhoFm] = (rho * yF).squeeze(axis=2)

		# Combine 1D water layer and 1D magma
		Uq = np.reshape(
			np.where(x[...,-1].ravel()[...,np.newaxis] >= self.z_vent,
				np.reshape(Uq_water, (-1, physics.NUM_STATE_VARS)), # Water layer
				np.where(x[...,-1].ravel()[...,np.newaxis] >= self.z_premix, # Premix porous layer
					np.reshape(Uq_premix, (-1, physics.NUM_STATE_VARS)),
					np.reshape(Uq_magma, (-1, physics.NUM_STATE_VARS)))),
			Uq.shape)

		return Uq # [ne, nq, ns]


class _IsothermalAtmosphere_LavaLake(FcnBase):
	'''
	Isothermal air atmosphere as an initial condition. LavaLake comments, Unused
	'''

	# Use 15e6 p_h0 for water (WLMA~11)
	def __init__(self,T:float=300., p_h0=35e6, # # p_h0:float=15e6, #1e3*1300*9.8,
		h0:float=-800.0, hmax:float=2500.0, gravity:float=9.8,
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

		# Determine mass fractions for seawater composition
		# TODO: rework passing in composition
		self.y_underwater = np.array([self.tracer_frac,
			1.0-2.0*self.tracer_frac, self.tracer_frac])
		self.y_overwater = np.array([1.0-self.tracer_frac-1e-7, # TODO: update water trace content
			1e-7, self.tracer_frac])

		# Lava lake setting
		# self.y_underwater = np.array([self.tracer_frac,
		# 	0.0, 1.0-self.tracer_frac])
		# self.y_overwater = np.array([1.0-self.tracer_frac,
		# 	0.0, self.tracer_frac])

		# Renormalize out air (dependent)
		if self.y_underwater[0] < 0:
			self.y_underwater[0] = 0
			self.y_underwater /= self.y_underwater.sum()
		# Define evaluation points from input mesh y-coordinate
		eval_pts = np.unique(x[:,:,1:2])

		def diffuse_y(p:float):
			'''Diffuse boundary over [p_surface, 2*p_surface] (length scale of
			  diffuse_factor * p_surface / (rho * g) ~ diffuse_factor * (10 m)
			  for liquid water)'''
			# y = self.y_underwater if p > p_surface else self.y_overwater
			diffuse_factor = .02
			theta = np.clip( (p - p_surface) / (diffuse_factor * p_surface), 0, 1.0)
			y = theta * self.y_underwater + (1 - theta) * self.y_overwater
			return y

		p_surface = 5e5
		def mixture_spec_vol(height:float, p:float, T:float):
			# Specific volume as function of p, T
			# Select composition
			# T = 700 if height > 200 else T
			y = diffuse_y(p)

			# Volume and composition of water layer
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		# Evaluate IVP solution
		soln = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / mixture_spec_vol(height, p, self.T),
			[self.h0, self.hmax],
			[self.p_h0],
			t_eval=eval_pts,
			dense_output=True)
		# Cache the pressure interpolant
		self.pressure_interpolant = soln.sol
		# Use equivalent height for simulating uplift
		uplift = lambda r: 300.0 * np.exp(-((r-0)/200.0)**2)
		y_equiv = x[...,1] - uplift(x[...,0])
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p = np.reshape(self.pressure_interpolant(y_equiv.ravel()), y_equiv.shape)
		# Compute resultant density
		rho = np.reshape(
			1.0 / np.array([mixture_spec_vol(_z, _p, self.T)
				for (_z, _p) in zip(y_equiv.ravel(), p.ravel())]),
			p.shape)
		
		# # Compute partial densities for underwater part
		# arhoA  = self.y_underwater[0] * rho
		# arhoWv = self.y_underwater[1] * rho
		# arhoM  = self.y_underwater[2] * rho
		# # Compute partial densities for overwater part
		# arhoA[p <= p_surface]  = self.y_overwater[0] * rho[p <= p_surface]
		# arhoWv[p <= p_surface] = self.y_overwater[1] * rho[p <= p_surface]
		# arhoM[p <= p_surface]  = self.y_overwater[2] * rho[p <= p_surface]
		# Compute diffuse mass variables
		arhoA = rho * np.reshape(np.array([diffuse_y(_p)[0] for _p in p.ravel()]), p.shape)
		arhoWv = rho * np.reshape(np.array([diffuse_y(_p)[1] for _p in p.ravel()]), p.shape)
		arhoM = rho * np.reshape(np.array([diffuse_y(_p)[2] for _p in p.ravel()]), p.shape)

		# Compute conservative variables for tracers (typically no source in 2D)
		arhoWt = arhoWv + self.tracer_frac * arhoM
		arhoC  = self.tracer_frac * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM
		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute specific energy
		e_m = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p.ravel()]), p.shape)
		# Variable temperature mod
		# T = np.zeros_like(p)
		# T[:] = np.where(x[...,1] > 200, 700, self.T)
		# e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
		# 	mixtureWLMA.float_mix_functions.rho_l_pt(_p, _T), _T)
		# 	for (_p, _T) in zip(p.ravel(), T.ravel())]), p.shape)
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, self.T), self.T)
			for _p in p.ravel()]), p.shape)
		rho = arhoA + arhoWv + arhoM
		# Compute volumetric energy
		# TODO: move implementation to middleware
		e = (arhoA * physics.Gas[0]["c_v"] * self.T + 
				arhoWv * e_w + 
				arhoM * (physics.Liquid["c_m"] * self.T + e_m)
				+ 0.5 * rho * (u*u+v*v)) # TODO: vpT analog needs correction for uv kinetc
		# Assemble conservative state vector		
		Uq[:, :, iarhoA]  = arhoA
		Uq[:, :, iarhoWv] = arhoWv
		Uq[:, :, iarhoM]  = arhoM
		Uq[:, :, irhou]   = rho * u
		Uq[:, :, irhov]   = rho * v
		Uq[:, :, ie]      = e
		# Tracer quantities
		Uq[:, :, iarhoWt] = arhoWt
		Uq[:, :, iarhoC]  = arhoC
		Uq[:, :, iarhoFm] = arhoFm

		return Uq # [ne, nq, ns]

class IsothermalAtmosphere1D(FcnBase):
	'''
	1D slice of isothermal air atmosphere as an initial condition.
	'''

	# Use 15e6 p_h0 for water (WLMA~11)
	def __init__(self,T:float=300., p_h0=35e6, pchamber=100e6, # # p_h0:float=15e6, #1e3*1300*9.8,
		hchamber = -1000, hjump=-350,
		h0:float=-800.0, hmax:float=2500.0, gravity:float=9.8,
		massFracWv:float=1.0-1e-7, massFracM:float=1e-7, tracer_frac:float=1e-7):
		''' Set atmosphere temperature, pressure, and location of pressure.
		Pressure distribution is computed as hydrostatic profile with p = p_h0
		at elevation h0.
		'''
		self.T = T
		self.p_h0 = p_h0
		self.h0 = h0
		self.hmax = hmax
		self.hchamber = hchamber
		self.hjump = hjump
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
		iarhoA, iarhoWv, iarhoM, irhou, ie, iarhoWt, iarhoC, iarhoFm = \
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

		''' Compute hydrostatic pressure as initial value problem (IVP)
		with two-part composition: y_low and y_high. '''

		self.y_low = np.array([self.tracer_frac,
													 self.waterFrac * (1.0 - self.tracer_frac),
													 (1.0 - self.waterFrac) * (1.0 - self.tracer_frac)])
		self.y_high = np.array([self.tracer_frac,
													  1.0 - 2 * self.tracer_frac,
														self.tracer_frac])
		# Define evaluation points for ode solver
		eval_pts = np.unique(x[:,:,0:1])

		# def diffuse_y(p:float):
		# 	'''Diffuse boundary over [p_surface, 2*p_surface] (length scale of
		# 	  diffuse_factor * p_surface / (rho * g) ~ diffuse_factor * (10 m)
		# 	  for liquid water)'''
		# 	# y = self.y_underwater if p > p_surface else self.y_overwater
		# 	diffuse_factor = .02
		# 	theta = np.clip( (p - p_surface) / (diffuse_factor * p_surface), 0, 1.0)
		# 	y = theta * self.y_underwater + (1 - theta) * self.y_overwater
		# 	return y

		def mixture_spec_vol(height:float, p:float, T:float, y:np.array) -> float:
			# Specific volume as function of p, T
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		# Evaluate pressure at self.hmax from upper IVP
		p_hmax = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / mixture_spec_vol(height, p, self.T, self.y_high),
			[self.h0, self.hmax],
			[self.p_h0],
			t_eval=[self.hmax],
			dense_output=False).y.ravel()[0]
		# Evaluate upper IVP solution, above jump position
		soln_high = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / mixture_spec_vol(height, p, self.T, self.y_high),
			[self.hmax, self.hjump],
			[p_hmax],
			t_eval=eval_pts[eval_pts >= self.hjump],
			dense_output=True)
		# Evaluate lower IVP solution, below jump position
		soln_low = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / mixture_spec_vol(height, p, self.T, self.y_low),
			[self.hchamber, self.hjump],
			[self.pchamber],
			t_eval=eval_pts[eval_pts <= self.hjump],
			dense_output=True)
		# Construct pressure interpolant
		pressure_interpolant = lambda x: np.piecewise(x,
						[x <= self.hjump, x > self.hjump],
						[soln_low.sol, soln_high.sol])
		density_interpolant = lambda x: np.piecewise(x,
						[x <= self.hjump, x > self.hjump],
						[mixture_spec_vol.sol, soln_high.sol])
		
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p = np.reshape(pressure_interpolant(x.ravel()), x.shape)
		# Compute resultant density
		rho = np.reshape(
			1.0 / np.array([
				mixture_spec_vol(_z, _p, self.T, self.y_low) if _z <= self.hjump
				else mixture_spec_vol(_z, _p, self.T, self.y_high)
				for (_z, _p) in zip(x.ravel(), p.ravel())]),
			p.shape)
		
		# Compute mass variables
		arhoA = rho * np.reshape(np.array([self.y_low[0] if _z <= self.hjump
						else self.y_high[0] for _z in x.ravel()]), p.shape)
		arhoWv = rho * np.reshape(np.array([self.y_low[1] if _z <= self.hjump
						else self.y_high[1] for _z in x.ravel()]), p.shape)
		arhoM = rho * np.reshape(np.array([self.y_low[2] if _z <= self.hjump
						else self.y_high[2] for _z in x.ravel()]), p.shape)

		# Compute conservative variables for tracers (typically no source in 2D)
		sol = self.physics.Solubility["k"] * p ** self.physics.Solubility["n"]
		arhoWt = arhoWv + sol / (1.0 + sol) * arhoM
		arhoC  = self.tracer_frac * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM
		# Zero velocity
		u = np.zeros_like(p)
		# Compute specific energy
		e_m = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p.ravel()]), p.shape)
		# Variable temperature mod
		# T = np.zeros_like(p)
		# T[:] = np.where(x[...,1] > 200, 700, self.T)
		# e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
		# 	mixtureWLMA.float_mix_functions.rho_l_pt(_p, _T), _T)
		# 	for (_p, _T) in zip(p.ravel(), T.ravel())]), p.shape)
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, self.T), self.T)
			for _p in p.ravel()]), p.shape)
		rho = arhoA + arhoWv + arhoM
		# Compute volumetric energy
		# TODO: move implementation to middleware
		e = (arhoA * physics.Gas[0]["c_v"] * self.T + 
				arhoWv * e_w + 
				arhoM * (physics.Liquid["c_m"] * self.T + e_m)
				+ 0.5 * rho * (u*u)) # TODO: vpT analog needs correction for uv kinetc
		# Assemble conservative state vector		
		Uq[:, :, iarhoA]  = arhoA
		Uq[:, :, iarhoWv] = arhoWv
		Uq[:, :, iarhoM]  = arhoM
		Uq[:, :, irhou]   = rho * u
		Uq[:, :, ie]      = e
		# Tracer quantities
		Uq[:, :, iarhoWt] = arhoWt
		Uq[:, :, iarhoC]  = arhoC
		Uq[:, :, iarhoFm] = arhoFm

		return Uq # [ne, nq, ns]

class DebrisFlow(FcnBase):
	'''
	Isothermal air atmosphere as an initial condition.
	'''

	def __init__(self,T:float=300., p_h0:float=30e6, #1e3*1300*9.8,
		h0:float=-800.0, hmax:float=2500.0, gravity:float=9.8,
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

		# Determine mass fractions for seawater composition
		self.y_underwater = np.array([1.0-self.massFracWv-self.massFracM,
			self.massFracWv, self.massFracM])
		self.y_debris = np.array([self.tracer_frac,
			self.tracer_frac, 1.0 - 2.0*self.tracer_frac])
		# Renormalize out air (dependent)
		if self.y_underwater[0] < 0:
			self.y_underwater[0] = 0
			self.y_underwater /= self.y_underwater.sum()
		# Define evaluation points from input mesh y-coordinate
		eval_pts = np.unique(x[:,:,1:2])

		p_surface = 5e5
		def mixture_spec_vol(height:float, p:float, T:float):
			# Specific volume as function of p, T
			# Select composition
			# T = 700 if height > 200 else T
			y = self.y_underwater
			# Volume and composition of water layer
			return y[0] * physics.Gas[0]["R"] * T / p \
				+ y[1] / mixtureWLMA.float_mix_functions.rho_l_pt(p, T) \
				+ y[2] / (physics.Liquid["rho0"] 
					* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"]))

		# Evaluate IVP solution
		soln = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / mixture_spec_vol(height, p, self.T),
			[self.h0, self.hmax],
			[self.p_h0],
			t_eval=eval_pts,
			dense_output=True)
		# Cache the pressure interpolant
		self.pressure_interpolant = soln.sol
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p = np.reshape(self.pressure_interpolant(x[...,1].ravel()), x[...,1].shape)
		# Compute resultant density
		rho = np.reshape(
			1.0 / np.array([mixture_spec_vol(_z, _p, self.T)
				for (_z, _p) in zip(x[...,1].ravel(), p.ravel())]),
			p.shape)
		# Compute partial densities for underwater part
		arhoA  = self.y_underwater[0] * rho
		arhoWv = self.y_underwater[1] * rho
		arhoM  = self.y_underwater[2] * rho
		# Replace section
		column_indices = (np.abs(x[...,0]) < 200.0) & (x[...,1] < 200.0)
		arhoM[column_indices] = 0.99999 * (physics.Liquid["rho0"] 
			* (1.0 + (p[column_indices] - physics.Liquid["p0"])/physics.Liquid["K"]))
		arhoA[column_indices] = self.tracer_frac
		arhoWv[column_indices] = 0.00001 * 1000
		
		# Compute conservative variables for tracers (typically no source in 2D)
		arhoWt = arhoWv + self.tracer_frac * arhoM
		arhoC  = self.tracer_frac * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM
		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute specific energy
		e_m = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p.ravel()]), p.shape)
		# Variable temperature mod
		T = self.T * np.ones_like(p)
		# T[:] = np.where(x[...,1] > 200, 700, self.T)
		T[column_indices] = 300
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, _T), _T)
			for (_p, _T) in zip(p.ravel(), T.ravel())]), p.shape)
		# e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
		# 	mixtureWLMA.float_mix_functions.rho_l_pt(_p, self.T), self.T)
		# 	for _p in p.ravel()]), p.shape)
		rho = arhoA + arhoWv + arhoM
		# Compute volumetric energy
		# TODO: move implementation to middleware
		e = (arhoA * physics.Gas[0]["c_v"] * T + 
				arhoWv * e_w + 
				arhoM * (physics.Liquid["c_m"] * T + e_m)
				+ 0.5 * rho * (u*u+v*v)) # TODO: vpT analog needs correction for uv kinetc
		# Assemble conservative state vector		
		Uq[:, :, iarhoA]  = arhoA
		Uq[:, :, iarhoWv] = arhoWv
		Uq[:, :, iarhoM]  = arhoM
		Uq[:, :, irhou]   = rho * u
		Uq[:, :, irhov]   = rho * v
		Uq[:, :, ie]      = e
		# Tracer quantities
		Uq[:, :, iarhoWt] = arhoWt
		Uq[:, :, iarhoC]  = arhoC
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
class LinearizedImpedance2D(BCWeakRiemann):
	'''
	Impedance boundary condition linearized about the initial pressure
	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, p_source:float=25e6, T_source:float=1000.0, u:float=100.0):
		''' Pressure is initialized using the first encountered pressure. '''
		self.initialized = False
		# Extract specified mass flux and prescribed chamber/reservoir values
		self.p_source, self.T_source, self.u = p_source, T_source, u

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly in
		a linearized sense. '''

		UqB = np.zeros_like(UqI)
		# ''' Check validity of flow state, check number of boundary points. '''
		# if UqI.shape[0] * UqI.shape[1] > 1:
		# 	raise NotImplementedError('''Not implemented: for-loop over more than one
		# 		inflow boundary point.''')
		
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		
		# import scipy.optimize
		p_tune = 30e6
		T = 1000
		# rhow_tuned = scipy.optimize.fsolve(
		# 	lambda rhow: mixtureWLMA.float_phi_functions.p(rhow ,T) - p_tune,
		# 	60)
		rhow_design = 70.4202492860682 # 30 MPa, 1000 K
		
		# rhow_design = 45.695633185787855 						 # 20 MPa, 1000 K
		rhoa_design = p_tune / (physics.Gas[0]["R"] * T)
		rhom_design = rho0 * (1.0 + (p_tune - p0) / K) # 20 MPa
		ya_design = 1e-7
		yw_design = 0.05
		ym_design = 0.0 # 1.0 - (ya_design + yw_design)
		v_design = (ya_design / rhoa_design 
	      + yw_design / rhow_design 
				+ ym_design / rhom_design)
		rho_design = 1.0 / v_design
		# Specific energy
		e_design = ya_design * physics.Gas[0]["c_v"] * T \
		  + yw_design * mixtureWLMA.float_phi_functions.u(rhow_design, T) \
			+ ym_design * (physics.Liquid["c_m"] * T
		    + mixtureWLMA.float_mix_functions.magma_mech_energy(p_tune,
							  physics.Liquid["K"],
								physics.Liquid["p0"],
								physics.Liquid["rho0"]))
		c = mixtureWLMA.float_mix_functions.mix_sound_speed(rhow_design, p_tune, T,
      {
        "vol_energy": rho_design * e_design,
        "rho_mix": rho_design,
        "yw": yw_design,
        "ya": ya_design,
        "K": physics.Liquid["K"],
        "p_m0": physics.Liquid["p0"],
        "rho_m0": physics.Liquid["rho0"],
        "c_v_m0": physics.Liquid["c_m"],
        "R_a": physics.Gas[0]["R"],
        "gamma_a": physics.Gas[0]["gamma"],
      })
		# Set vertical velocity equal to entrance sound speed
		v = c
		# Get UqB
		UqB[:,:,0:1] = ya_design * rho_design
		UqB[:,:,1:2] = yw_design * rho_design
		UqB[:,:,2:3] = ym_design * rho_design
		UqB[:,:,3:4] = 0.0
		UqB[:,:,4:5] = rho_design * v
		UqB[:,:,5:6] = rho_design * e_design + 0.5 * rho_design * v * v
		return UqB


		''' END SHORT CIRCUIT '''

		UqB = UqI.copy()
		''' Compute normal velocity. '''
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		arhoVec = UqI[:,:,physics.get_mass_slice()]
		rhoGrid = arhoVec.sum(axis=-1, keepdims=True)
		velI = UqI[:, :, physics.get_momentum_slice()] / rhoGrid
		# Normal velocity scalar
		veln = np.sum(velI * n_hat, axis=2, keepdims=True)
		# Normal velocity vector
		velvec_n = np.einsum("...i, ...i -> ...i", veln, n_hat)
		# Tangent velocity vector
		velvec_t = velI - velvec_n

		''' Inflow handling '''
		# if np.any(velI < 0.): # TODO:
			# print("Incoming flow at outlet")

		''' Record first pressure encountered at boundary. '''
		if not self.initialized:
			self.p = physics.compute_variable("Pressure", UqI)
			self.initialized = True

		''' Compute interior quantities. '''
		# Call mixture backend
		rhow, pGrid, TGrid, cGrid, volfracW = \
			physics.wlma(
				UqI[:,:,physics.get_mass_slice()],
				UqI[:,:,physics.get_momentum_slice()],
				UqI[:,:,physics.get_state_slice("Energy")])
		pGrid = physics.compute_variable("Pressure", UqI)
		ZGrid = rhoGrid * cGrid
		uGrid = veln

		# Extract specified mass flux and prescribed chamber/reservoir values
		p_chamber, T_chamber = self.p_chamber, self.T_chamber
		# Extract material properties
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		# Approximate desired mass flux
		u = self.u

		# Compute composition-based properties
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# Outflow
			y = arhoVec / rhoGrid
			# TODO: properly entropy
			S = TGrid # TGrid / pGrid**((Gamma-1)/Gamma)
		else:
			# Inflow
			# arhoVecB = arhoVec.copy()
			# arhoVecB[...,0] = 0.0 * self.trace_arho # Small amount of air to preserve positivity
			# arhoVecB[...,1] = self.trace_arho # Small amount of water to preserve positivity
			# # Approximate partial density of magma by density
			# arhoVecB[...,2] = rho0 * (1.0 + (p_chamber - p0) / K)
			# y = arhoVecB / arhoVecB.sum(axis=-1, keepdims=True)
			y = np.zeros_like(arhoVec / rhoGrid)
			y[...,1:2] = 0.05
			y[...,2:3] = 0.95
			# TODO: properly entropy
			S = self.T_source # T_chamber / p_chamber**((Gamma-1)/Gamma)

		# Evaluate primitive variables for boundary state
		p = pGrid - (rhoGrid * cGrid) * (uGrid - u)
		T = S # * p**((Gamma-1)/Gamma)
		rho = atomics.mixture_density(y, p, T, physics)

		''' Check positivity of computed state. '''
		if np.any(T < 0.) or np.any(p < 0) or np.any(rho < 0):
			raise errors.NotPhysicalError

		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho * y
		UqB[:,:,physics.get_momentum_slice()] = rho * u
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho * y, physics) * T \
			+ (rho * y[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * rho * u * u

		# TODO: self.yWt, self.yC are not used

		# Update adiabatically compressed/expanded tracer partial densities
		UqB[:,:,5:] *= rho / atomics.rho(arhoVecI)
		# crystal vol / suspension vol
		phi_crys = 0.4 * (1.1 - 0.1 * np.cos(2 * np.pi * self.freq * t))
		chi_water = 0.03
		UqB[:,:,5] = rho * chi_water / (1 + chi_water) \
			* (1.0 - 0.4 * (1.1 - 0.1 * np.cos(2 * np.pi * self.freq * 0.0)))  # frozen
		UqB[:,:,6] = rho * phi_crys
	
		# Fragmented state
		UqB[:,:,7] = 0.0

		return UqB

		''' Relict of LinImp2D '''

		psiPlus = uGrid + (pGrid-self.p) / ZGrid
		uHat = 0.5 * psiPlus
		pHat = self.p + ZGrid * (0.5 * psiPlus)
		# Isentropic temp change
		Gamma = atomics.Gamma(UqI[...,0:3], physics)
		THat = TGrid * (pHat/pGrid)**((Gamma-1)/Gamma)

		if np.any(pHat < 0.):
			raise errors.NotPhysicalError

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


class PressureStableLinearizedInlet1D(BCWeakPrescribed):
	'''
	See physics/multiphasevpT/functions.py > PressureStableLinearizedInlet1D
	for the function this is based on. Here the equation of state is changed,
	and the isothermal approximation is used for computing the boundary
	temperature.
	'''
	def __init__(self, p_chamber:float=100e6, T_chamber:float=1e3, trace_arho:float=2.6e-4,
			chi_water:float=0.03, cVFav:float=0.4, cVFamp:float=0.25, is_gaussian:bool=False,
			cos_freq:float=0.0, gaussian_tpulse:float=20.0, gaussian_sig:float=4.0):
		# Arguments for chamber properties
		self.p_chamber, self.T_chamber, self.trace_arho = \
			p_chamber, T_chamber, trace_arho
		# Water concentration
		self.chi_water = chi_water
		# Crystal volume fraction average and perturbation amplitude
		self.cVFav, self.cVFamp = cVFav, cVFamp
		# Gaussian pulse if true (else sinusoidal)
		self.is_gaussian:bool = is_gaussian
		# Frequency for sinusodial (used if not is_gaussian)
		self.cos_freq = cos_freq
		# Gaussian properties (used if is_gaussian)
		self.gaussian_tpulse, self.gaussian_sig = gaussian_tpulse, gaussian_sig

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Compute a boundary state by replacing Riemann problem with acoustic
		waves, and then approximating the acoustic wave. '''
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point in 1D.''')

		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to outgoing acoustic waves) of the interior
		solution. '''
		# Extract data from node values
		arhoVecI = UqI[:,:,physics.get_mass_slice()]
		momxI = UqI[...,physics.get_momentum_slice()]
		eI = UqI[...,physics.get_state_slice("Energy")]
		# Extract specified mass flux and prescribed chamber/reservoir values
		p_chamber, T_chamber = self.p_chamber, self.T_chamber
		# Extract material properties
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		
		# Compute grid primitives
		TGrid = physics.compute_variable("Temperature", UqI)
		pGrid = physics.compute_variable("Pressure", UqI)
		rhoGrid = UqI[...,0:3].sum(axis=-1, keepdims=True)
		cGrid = physics.compute_variable("SoundSpeed", UqI)
		uGrid = momxI / rhoGrid
		# Compute chamber mass properties
		arhoVecChamber = arhoVecI.copy()
		arhoVecChamber[...,0] = self.trace_arho # Small amount of air to preserve positivity
		arhoVecChamber[...,1] = self.trace_arho # Small amount of water to preserve positivity
		# Approximate partial density of magma by density
		arhoVecChamber[...,2] = rho0 * (1 + (p_chamber - p0) / K)
		rhoChamber = arhoVecChamber.sum(axis=-1, keepdims=True)
		# Use approximate sound speed for chamber (assume no exsolved water)
		cChamber = np.sqrt(K / rho0)

		# Compute composition-based properties
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals >= 0.):
			# Outflow
			y = atomics.massfrac(arhoVecI)
			T = TGrid
			rho_base = rhoGrid
			c_base = cGrid
			p_base = pGrid
		else:
			# Inflow
			y = atomics.massfrac(arhoVecChamber)
			T = T_chamber
			rho_base = rhoChamber
			c_base = cChamber
			p_base = self.p_chamber

		# Compute acoustic impedances Z and admittances Y = 1/Z
		ZGrid = rhoGrid * cGrid
		ZChamber = rhoChamber * cChamber
		YGrid = 1.0 / ZGrid
		YChamber = 1.0 / ZChamber
		# Approximate pressure as Y-weighted average
		p = (YGrid * pGrid + YChamber * p_chamber) / (YGrid + YChamber)
		# Approximate corresponding velocity in x-direction
		u = uGrid + (p_chamber - pGrid) / (ZChamber + ZGrid)

		# Evaluate density from first-order perturbation from upwind s
		rho = rho_base + (p - p_base)/ (c_base * c_base)

		''' Check positivity of computed state. '''
		if np.any(T < 0.) or np.any(p < 0) or np.any(rho < 0):
			raise errors.NotPhysicalError
		
		# Compute energy
		e_a = physics.Gas[0]["c_v"] * T
		e_m_mech = np.reshape(np.array([mixtureWLMA.float_mix_functions.magma_mech_energy(_p,
			physics.Liquid["K"], physics.Liquid["p0"], physics.Liquid["rho0"])
			for _p in p.ravel()]), p.shape)
		e_m = physics.Liquid["c_m"] * T + e_m_mech
		# Water energy, assuming supercritical (can use rho_l for this case)
		e_w = np.reshape(np.array([mixtureWLMA.float_mix_functions.u(
			mixtureWLMA.float_mix_functions.rho_l_pt(_p, T), T)
			for _p in p.ravel()]), p.shape)		
		e_vec = np.concatenate((np.asarray(e_a).ravel(), np.asarray(e_w).ravel(), np.asarray(e_m).ravel(),), )
		e_mix = (y.ravel() * e_vec).sum()

		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho * y
		UqB[:,:,physics.get_momentum_slice()] = rho * u
		UqB[:,:,physics.get_state_slice("Energy")] = rho * e_mix
		# Update adiabatically compressed/expanded tracer partial densities
		UqB[:,:,5:] *= rho / atomics.rho(arhoVecI)

		# crystal vol / suspension vol
		ta = 5
		tb = 5 + (1 / (2 * self.cos_freq)) if self.cos_freq != 0 else +np.inf
		if self.is_gaussian:
			phi_crys = self.cVFav + self.cVFamp \
				* np.exp(-((t - self.gaussian_tpulse)/ self.gaussian_sig) **2 / 2)
		else:
			if t < ta:
				phi_crys = self.cVFav
			elif t < tb:
				halfAmp = self.cVFamp / 2
				phi_crys = self.cVFav * ((1-halfAmp) + halfAmp * (np.cos(2 * np.pi * self.cos_freq * (t - ta)) - 1) / 2)
			else:
				phi_crys = self.cVFav * (1 - self.cVFamp * np.cos(2 * np.pi * self.cos_freq * (t - tb)))
		chi_water = self.chi_water
		UqB[:,:,5] = rho * chi_water * (1.0 - phi_crys) / (1 + chi_water)
		UqB[:,:,6] = rho * phi_crys
		# Fragmented state
		UqB[:,:,7] = 0.0
	
		return UqB


class WaterMassSource(SourceBase):
	'''
	Added water mass source. Distributes a mass rate over a Gaussian function.
	Partial density rate is used instead if partial_density_rate is not None.
	  mass_rate: sets the rate of water added in units of mass per time.
		specific_energy: sets prescribed specific energy (J/kg) of influx water

	Note: the density rate is related to the total mass rate by
		mass_rate = int (partial_density_rate * pi * conduit_radius**2) dz,
	meaning that int (partial_density_rate) dz = mass_rate / (pi * radius**2).

	Default specific energy is computed for:
	  p = 4.0000e7 Pa, T = 300 K
	giving
	  rho = 1013.7496496496497 kg/m^3
	and
	  e = 109388.56885035457 J/kg,
	where e is relative to the energy of pure liquid at the triple point.
	'''
	def __init__(self,
							 mass_rate_per_depth=1e5,
							 specific_energy=109388.56885035457,
							 injection_depth=-350,
							 gaussian_width=50,
							 conduit_radius=25,
							 hwm=-50,
							 injection_time_start=0.0,
							 injection_time_end=2.0,
							 **kwargs):
		super().__init__(kwargs)
		self.mass_rate_per_depth = mass_rate_per_depth
		self.specific_energy = specific_energy
		self.injection_depth = injection_depth
		self.gaussian_width = gaussian_width
		self.conduit_radius = conduit_radius
		self.hwm = hwm
		self.injection_time_start = injection_time_start
		self.injection_time_end = injection_time_end

		self._sqrtpi = np.sqrt(np.pi)

	def scaled_gaussian(self, x):
		''' Returns the scaled and shifted gaussian with unit integral. '''
		_t = (x - self.injection_depth) / self.gaussian_width
		return np.exp(-_t*_t) / (self._sqrtpi * self.gaussian_width)

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		# Compute mixture density
		state_indices = physics.get_state_indices()
		if len(state_indices) == 8:
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
					physics.get_state_indices()
		elif len(state_indices) == 9:
			iarhoA, iarhoWv, iarhoM, imomx, imomy, ie, iarhoWt, iarhoC, iarhoFm = \
					physics.get_state_indices()
		else:
			raise ValueError("Unknown number of state_indices; can't unpack state_indices")
		# Get mass-specific energy from input
		specific_energy = self.specific_energy
		# Compute partial density rate scaling
		partial_density_rate = self.mass_rate_per_depth / (
			np.pi * self.conduit_radius * self.conduit_radius)
		
		# Compute partial density source using last spatial coordinate (x in 1D, y in 2D) as vertical spatial coordinate
		partial_density_source = partial_density_rate * self.scaled_gaussian(x[:,:,-1:])
		# Time-limit and space-limit
		partial_density_source = np.where(x[:,:,-1:] >= self.hwm,
																		  0.0,
																			partial_density_source)
		partial_density_source *= float(t >= self.injection_time_start
																		and t <= self.injection_time_end)

		# Assemble and return source vector
		S[:,:,iarhoWv:iarhoWv+1] = partial_density_source
		S[:,:,iarhoWt:iarhoWt+1] = partial_density_source
		S[:,:,ie:ie+1]=  partial_density_source * specific_energy
		return S # [ne, nq, ns]

class WaterEntrainmentSource(SourceBase):
	'''
	Added water mass source that scalings with quasi-1D velocity.
		entrainment_coeff: coefficient of proportionality mass rate per velocity
		specific_energy: sets prescribed specific energy (J/kg) of influx water

	Note: the density rate is related to the total mass rate by
		mass_rate = int (partial_density_rate * pi * conduit_radius**2) dz,
	meaning that int (partial_density_rate) dz = mass_rate / (pi * radius**2).
	'''
	def __init__(self,
							 entrainment_coefficient=1e-2,
							 T_ref=300,
							 p_ref=1e5,
							 injection_depth=-350,
							 smoothing_scale=50,
							 conduit_radius=50,
							 theta_deg=10, # Jet angle in degrees
							 h1=-150,      # Jet nozzle depth
							 **kwargs):
		super().__init__(kwargs)
		self.entrainment_coefficient = entrainment_coefficient
		self.T_ref = T_ref
		self.p_ref = p_ref
		self.injection_depth = injection_depth
		self.smoothing_scale = smoothing_scale
		self.conduit_radius = conduit_radius
		self.theta_deg = theta_deg
		self.h1 = h1

		# Initial thermodynamic computation: compute energy at reference pressure
		# Energy is not sensitive to pressure perturbation
		solnobj = scipy.optimize.root_scalar(
			lambda rho: mixtureWLMA.float_phi_functions.p(rho, 300) / p_ref - 1,
			x0=996.5,
			x1=997)
		if not solnobj.converged:
			raise ValueError(f"Failed to compute water density. Dumping root finding object: {solnobj}")
		self.rhow = solnobj.root
		self.specific_energy = mixtureWLMA.float_phi_functions.u(self.rhow, T_ref)

	def get_radius(self, x):
		''' Returns radius of conduit or underwater jet. '''
		# Interpret h1 as quasi-1D jet nozzle location
		h1 = self.h1
		# Compute radius
		return np.where(x > h1,
							self.conduit_radius + (x - h1) * np.tan(self.theta_deg * np.pi / 180),
							self.conduit_radius)

	def scaled_gaussian(self, x):
		''' Legacy code. Returns the scaled and shifted gaussian with unit integral. '''
		_t = (x - self.injection_depth) / self.gaussian_width
		return np.exp(-_t*_t) / (self._sqrtpi * self.gaussian_width)
	
	def one_side_bump(self, x):
		''' Smoothed Heaviside function H(x) with properties
		1. H(x) = 0, x < self.injection_depth
		2. H(x) = 1, x > self.injection_depth + self.smoothing_scale
		3. H is smooth
		'''
		# Shift, scale, and clip to [-1, 0] to prevent exp overflow
		_x = np.clip((x-(self.injection_depth + self.smoothing_scale))
							    / self.smoothing_scale + 1, 0, 1)
		f0 = np.exp(-1/np.where(_x == 0, 1, _x))
		f1 = np.exp(-1/np.where(_x == 1, 1, 1-_x))
		# Return piecewise evaluation
		return np.where(_x >= 1, 1,
						np.where(_x <= 0, 0, 
							f0 / (f0 + f1)))

	def get_source(self, physics, Uq, x, t):
		# Allocate
		S = np.zeros_like(Uq)
		if physics.NDIMS > 1:
			raise ValueError(f"WaterEntrainmentSource is appropriate for 1 dimension only, but got {physics.NDIMS} dimensions.")
		# Unpack indices
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		rho = Uq[...,0:3].sum(axis=-1, keepdims=True)
		u = Uq[...,imom:imom+1]	/ rho
		# Get mass-specific energy from input
		specific_energy = self.specific_energy
		# Compute surface flux as (kg / (m^2 * s))
		surface_flux = self.entrainment_coefficient * np.abs(u) * self.rhow # * self.one_side_bump(x[:,:,-1:])
		# Compute 4/D_H coefficient
		coeff = np.where(x > self.h1,
									 	 2.0 / self.get_radius(x),
										 0.0)
		# Compute partial density rate from surface flux (4 / D_h == 2 / R)
		partial_density_rate = coeff * surface_flux		
		# Assemble and return source vector
		S[:,:,iarhoWv:iarhoWv+1] = partial_density_rate
		S[:,:,iarhoWt:iarhoWt+1] = partial_density_rate
		S[:,:,ie:ie+1]=  partial_density_rate * specific_energy
		return S # [ne, nq, ns]

class MagmaMassSource(SourceBase):
	'''
	Added magma mass source. Distributes a mass rate over a Gaussian function.
	Partial density rate is used instead if partial_density_rate is not None.
	  mass_rate: sets the rate of water added in units of mass per time.
		specific_energy: sets prescribed specific energy (J/kg) of influx water

	Note: the density rate is related to the total mass rate by
		mass_rate = int (partial_density_rate * pi * conduit_radius**2) dz,
	meaning that int (partial_density_rate) dz = mass_rate / (pi * radius**2).
	'''
	def __init__(self,
							 mass_rate=0.0,
							 specific_energy=3e3*1000.0,
							 injection_depth=-350,
							 gaussian_width=50,
							 conduit_radius=50,
							 cutoff_height=-150,
							 **kwargs):
		super().__init__(kwargs)
		self.mass_rate = mass_rate
		self.specific_energy = specific_energy
		self.injection_depth = injection_depth
		self.gaussian_width = gaussian_width
		self.conduit_radius = conduit_radius
		self.cutoff_height = cutoff_height
		self._sqrtpi = np.sqrt(np.pi)

	def scaled_gaussian(self, x):
		''' Returns the scaled and shifted gaussian with unit integral,
		multiplied with a hard cutoff H(cutoff_height - x) '''
		_t = (x - self.injection_depth) / self.gaussian_width
		_cutoff = (np.asarray(x) < self.cutoff_height).astype(float)
		return np.exp(-_t*_t) / (self._sqrtpi * self.gaussian_width) * _cutoff

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		# Compute mixture density
		state_indices = physics.get_state_indices()
		if len(state_indices) == 8:
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
					physics.get_state_indices()
		elif len(state_indices) == 9:
			iarhoA, iarhoWv, iarhoM, imomx, imomy, ie, iarhoWt, iarhoC, iarhoFm = \
					physics.get_state_indices()
		else:
			raise ValueError("Unknown number of state_indices; can't unpack state_indices")
		# Get mass-specific energy from input
		specific_energy = self.specific_energy
		# Compute partial density rate scaling
		partial_density_rate = self.mass_rate / (
			np.pi * self.conduit_radius * self.conduit_radius)
		# Compute partial density source using last spatial coordinate (x in 1D, y in 2D)
		partial_density_source = partial_density_rate * self.scaled_gaussian(x[:,:,-1:])
		# Assemble and return source vector
		S[:,:,iarhoM:iarhoM+1] = partial_density_source
		S[:,:,ie:ie+1]=  partial_density_source * specific_energy
		return S # [ne, nq, ns]
	
class GravitySource(SourceBase):
	'''
	Clone of multiphasevpT GravitySource source term, with shut-off at given
	height to create a wave buffer region.
	
	Applies gravity for 1D in negative x-direction, and for 2D in negative
	y-direction.
	'''
	def __init__(self, gravity=0., cutoff_height=np.inf, **kwargs):
		super().__init__(kwargs)
		self.gravity = gravity
		self.cutoff_height = cutoff_height

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		# Compute mixture density
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2)
		if physics.NDIMS == 1:
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			g = np.where(x[:,:,-1] <= self.cutoff_height, self.gravity, 0.0)
			# Orient gravity in axial direction
			S[:, :, imom] = -rho * g
			S[:, :, ie]   = -Uq[:, :, imom] * g # rhou * g (gravity work)
		elif physics.NDIMS == 2:
			# Orient gravity in y direction
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			g = np.where(x[:,:,-1] <= self.cutoff_height, self.gravity, 0.0)
			S[:, :, irhov] = -rho * g
			S[:, :, ie] = -Uq[:, :, irhov] * g
		else:
			raise Exception("Unexpected physics num dimension in GravitySource.")
		return S # [ne, nq, ns]
	
	def get_jacobian(self, physics, Uq, x, t):
		jac = np.zeros([Uq.shape[0], Uq.shape[1], Uq.shape[-1], Uq.shape[-1]])
		if physics.NDIMS == 1:
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			# Orient gravity in axial direction
			jac[:, :, imom, [iarhoA, iarhoWv, iarhoM]] = -self.gravity
			jac[:, :, ie, imom] = -self.gravity
		elif physics.NDIMS == 2:
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			# Orient gravity in y-direction
			jac[:, :, irhov, [iarhoA, iarhoWv, iarhoM]] = -self.gravity
			jac[:, :, ie, irhov] = -self.gravity
		else:
			raise Exception("Unexpected physics num dimension in GravitySource.")
		return jac