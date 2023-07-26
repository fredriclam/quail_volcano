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
#       File : src/physics/multiphasevpT/functions.py
#
#       Contains definitions of Functions, boundary conditions, and source
#       terms for the multiphase vpT relaxation equations.
#
# ------------------------------------------------------------------------ #
from abc import abstractmethod
from enum import Enum, auto
import numpy as np
import scipy.integrate
import scipy.special as sp

import errors
import general
import logging

from physics.base.data import (BCBase, FcnBase, BCWeakRiemann, BCWeakPrescribed,
				SourceBase, ConvNumFluxBase)
import physics.multiphasevpT.atomics as atomics
import compressible_conduit_steady.steady_state as plugin_steady_state
from compressible_conduit_steady.advection_map import advection_map

from dataclasses import dataclass
import copy

class FcnType(Enum):
	'''
	Enum class that stores the types of analytical functions for initial
	conditions, exact solutions, and/or boundary conditions. These
	functions are specific to the available Euler equation sets.
	'''
	RiemannProblem = auto()
	GravityRiemann = auto()
	UniformExsolutionTest = auto()
	IsothermalAtmosphere = auto()
	LinearAtmosphere = auto()
	RightTravelingGaussian = auto()
	SteadyState = auto()
	NohProblem = auto()


class BCType(Enum):
	'''
	Enum class that stores the types of boundary conditions.
	'''
	SlipWall = auto()
	PressureOutlet = auto()
	Inlet = auto()
	CustomInlet = auto()
	MultiphasevpT1D1D = auto()
	MultiphasevpT2D1D = auto()
	MultiphasevpT2D2D = auto()
	MultiphasevpT2D1DCylindrical = auto()
	MultiphasevpT2D2DCylindrical = auto()
	NonReflective1D = auto()
	PressureOutlet1D = auto()
	PressureOutlet2D = auto()
	MassFluxInlet1D = auto()
	PressureStableLinearizedInlet1D = auto()
	VelocityInlet1D = auto()
	VelocityInlet1D_neutralSinusoid = auto()
	VelocityInlet1D_gaussianPulse = auto()
	LinearizedImpedance2D = auto()
	# Not implemented (could use lumped magma chamber model for example)
	EntropyTotalenthalpyInlet1D = auto()
	EntropyPressureInlet1D = auto()
	NohInlet = auto()
	LinearizedIsothermalOutflow2D = auto()


class SourceType(Enum):
	'''
	Enum class that stores the types of source terms. These
	source terms are specific to the available equation sets.
	'''
	FrictionVolFracVariableMu = auto()
	FrictionVolFracConstMu = auto()
	GravitySource = auto()
	ExsolutionSource = auto()
	FragmentationTimescaleSource = auto()
	FragmentationTimescaleSourceSmoothed = auto()
	WaterInflowSource = auto()
	CylindricalGeometricSource = auto()


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

class RiemannProblem(FcnBase):
	'''
	Riemann problem.

	Attributes:
	-----------
	rhoL: float
		left density
	uL: float
		left velocity
	pL: float
		left pressure
	rhoR: float
		right density
	uR: float
		right velocity
	pR: float
		right pressure
	xd: float
		location of initial discontinuity
	'''
	def __init__(self, arhoAL=1e-1, arhoWvL=8.686, arhoML=2496.3, uL=0.,
							 TL=1000., arhoWtL=10.0, arhoCL=1e-2, arhoFmL=1e-2,
							 arhoAR=1.161, arhoWvR=1.161*5e-3, arhoMR=1e-5, uR=0.,
							 TR=300., arhoWtR=1.161*5e-3, arhoCR=0.5e-5, xd=0., arhoFmR=0.5e-5):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			stuff
			xd: location of initial discontinuity

		Outputs:
		--------
				self: attributes initialized
		'''
		self.arhoAL = arhoAL
		self.arhoWvL = arhoWvL
		self.arhoML = arhoML
		self.uL = uL
		self.TL = TL
		self.arhoWtL = arhoWtL
		self.arhoCL = arhoCL
		self.arhoFmL = arhoFmL
		self.arhoAR = arhoAR
		self.arhoWvR = arhoWvR
		self.arhoMR = arhoMR
		self.uR = uR
		self.TR = TR
		self.xd = xd
		self.arhoWtR = arhoWtR
		self.arhoCR = arhoCR
		self.arhoFmR = arhoFmR

	def get_state(self, physics, x, t):
		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()
		arhoAL = self.arhoAL
		arhoWvL = self.arhoWvL
		arhoML = self.arhoML
		uL = self.uL
		TL = self.TL
		arhoWtL = self.arhoWtL
		arhoCL = self.arhoCL
		arhoFmL = self.arhoFmL
		arhoAR = self.arhoAR
		arhoWvR = self.arhoWvR
		arhoMR = self.arhoMR
		uR = self.uR
		TR = self.TR
		arhoWtR = self.arhoWtR
		arhoCR = self.arhoCR
		arhoFmR = self.arhoFmR

		rhoL = arhoAL+arhoWvL+arhoML
		eL = (arhoAL * physics.Gas[0]["c_v"] * TL + 
			arhoWvL * physics.Gas[1]["c_v"] * TL + 
			arhoML * (physics.Liquid["c_m"] * TL + physics.Liquid["E_m0"])
			+ 0.5 * rhoL * uL**2.)
		rhoR = arhoAR+arhoWvR+arhoMR
		eR = (arhoAR * physics.Gas[0]["c_v"] * TR + 
			arhoWvR * physics.Gas[1]["c_v"] * TR + 
			arhoMR * (physics.Liquid["c_m"] * TR + physics.Liquid["E_m0"])
			+ 0.5 * rhoR * uR**2.)

		for elem_ID in range(Uq.shape[0]):
			ileft = (x[elem_ID, :, 0] <= self.xd).reshape(-1)
			iright = (x[elem_ID, :, 0] > self.xd).reshape(-1)
			# Replacement to prevent quadratic approximation of in-element
			# discontinuity sending the state negative
			if np.all(x[elem_ID, :, 0] >= self.xd):
				ileft = (x[elem_ID, :, 0] < self.xd).reshape(-1)
				iright = (x[elem_ID, :, 0] >= self.xd).reshape(-1)
			# Fill left/right mass-related quantities
			Uq[elem_ID, ileft, iarhoA] = arhoAL
			Uq[elem_ID, iright, iarhoA] = arhoAR
			Uq[elem_ID, ileft, iarhoWv] = arhoWvL
			Uq[elem_ID, iright, iarhoWv] = arhoWvR
			Uq[elem_ID, ileft, iarhoM] = arhoML
			Uq[elem_ID, iright, iarhoM] = arhoMR
			# XMomentum
			Uq[elem_ID, ileft, imom] = rhoL*uL
			Uq[elem_ID, iright, imom] = rhoR*uR
			# Energy
			Uq[elem_ID, ileft, ie] = eL
			Uq[elem_ID, iright, ie] = eR
			# Tracer quantities
			Uq[elem_ID, ileft, iarhoWt] = arhoWtL
			Uq[elem_ID, iright, iarhoWt] = arhoWtR
			Uq[elem_ID, ileft, iarhoC] = arhoCL
			Uq[elem_ID, iright, iarhoC] = arhoCR
			Uq[elem_ID, ileft, iarhoFm] = arhoFmL
			Uq[elem_ID, iright, iarhoFm] = arhoFmR
		return Uq # [ne, nq, ns]

class RightTravelingGaussian(FcnBase):
	'''
	Gaussian in wave amplitude constructed to travel in the +x-direction.
	'''
	def __init__(self,
		p_ambient:float=1e5, T_ambient:float=300, y_ambient:np.array=None,
		amplitude:float=1.0, location:float=0.0, length_scale:float=30.0):
		'''
		Initialize with ambient pressure, temperature, and mass fraction
		vector (y). Also set the amplitude (in velocity units), the location,
		and the length scale of the initial Gaussian pulse.
		'''
		self.p_ambient = p_ambient
		self.T_ambient = T_ambient
		self.amplitude = amplitude
		self.location = location
		self.length_scale = length_scale
		if y_ambient is None:
			y_ambient = np.expand_dims(np.array([1.0, 0, 0]),axis=(0,1))
		elif len(y_ambient.shape) < 3:
			# Pad np.array to nd array matching shape of U
			y_ambient = np.zeros((1,1,1)) + y_ambient
		self.y_ambient = y_ambient

	def get_state(self, physics, x, t):
		U = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		# Compute 
		U[...,0] = 1.2
		U[...,1:3] = 0

		Gamma = atomics.Gamma(U[...,0:3], physics)
		gas_volfrac = atomics.gas_volfrac(U[...,0:3], self.T_ambient, physics)
		p = atomics.pressure(U[...,0:3], self.T_ambient, gas_volfrac, physics)

		# Construct constant psi+ with reference to psi+ == 0 at ambient state
		psi_plus = np.zeros_like(U[...,0:1])
		# Construct pulse in psi-
		psi_minus = self.amplitude / (
			np.sqrt(2.*np.pi)*self.length_scale) \
			* np.exp(-np.power((x - self.location)/self.length_scale, 2.)/2)
		
		u = 0.5*(psi_plus + psi_minus)
		# Compute linearized representation of mean impedance reciprocal
		f_bar = atomics.acousticRI_integrand_scalar(p[0,0,0], np.array([self.T_ambient]), p[0,0,0], 
			atomics.massfrac(U[...,0:3])[:1,:1,:], Gamma[0,0,0], physics)
		p = p[0,0,0] + 0.5*(psi_minus - psi_plus) / f_bar
		# Isentropic condition for temperature
		T = self.T_ambient * (p/p[0,0,0]) ** ((Gamma-1)/Gamma)

		# Fill in computed conservative state
		U[...,0:1] = p / T / physics.Gas[0]["R"]
		U[...,1:2] = 0*p / T / physics.Gas[1]["R"]
		U[...,2:3] = 0*p / T / physics.Gas[1]["R"]
		if np.any(U[...,2:3] > 0):
			raise NotImplementedError
		U[...,3:4] = atomics.rho(U[...,0:3]) * u
		U[...,4:5] = atomics.c_v(U[...,0:3], physics) * T \
		+ 0.5 * atomics.rho(U[...,0:3]) * u**2
		# Set all tracer quantities to zero
		U[...,5:] = 0
		return U # [ne, nq, ns]

class SteadyState(FcnBase):
	''' 1D steady state. Calls submodule compressible-conduit-steady
		(https://github.com/fredriclam/compressible-conduit-steady).
	'''
	def __init__(self, x_global:np.array=None, p_vent:float=1e5, inlet_input_val=1.0,
		input_type="u", yC=0.01, yWt=0.03, yA=1e-7, yWvInletMin=1e-5, yCMin=1e-5,
		crit_volfrac=0.7, tau_d=1.0, tau_f=1.0, conduit_radius=50,
		T_chamber=800+273.15, c_v_magma=3e3, rho0_magma=2.7e3, K_magma=10e9,
		p0_magma=5e6, solubility_k=5e-6, solubility_n=0.5, approx_massfracs=False,
		neglect_edfm=True, fragsmooth_scale=0.010):
		'''
		Interface to compressible_conduit_steady.SteadyState.
		'''
		if x_global is None:
			raise ValueError(
				"x_global must be specified in the SteadyState initial condition. " +
				"x_global provides 1D mesh information across all partitions of " +
				"the 1D domain.")
		props = {
				"yC": yC,
				"yWt": yWt,
				"yA": yA,
				"yWvInletMin": yWvInletMin,
				"yCMin": yCMin,
				"crit_volfrac": crit_volfrac,
				"tau_d": tau_d,
				"tau_f": tau_f,
				"conduit_radius": conduit_radius,
				"T_chamber": T_chamber,
				"c_v_magma": c_v_magma,
				"rho0_magma": rho0_magma,
				"K_magma": K_magma,
				"p0_magma": p0_magma,
				"solubility_k": solubility_k,
				"solubility_n": solubility_n,
				"neglect_edfm": neglect_edfm,
				"fragsmooth_scale": fragsmooth_scale,
		}
		self.approx_massfracs = approx_massfracs
		if approx_massfracs:
			# Pop off callable yC, yWt source time functions to match steady_state
			yC_fn = props.pop("yC")
			yWt_fn = props.pop("yWt")
			try:
				# Reattach initial guess yC, yWt as scalars
				props["yC"] = yC_fn(0.0)
				props["yWt"] = yWt_fn(0.0)
			except TypeError as e:
				raise TypeError("Flag approx_massfracs was set to True in initial " +
				"condition. This flag approximates the mass fraction distribution " +
				"in the initial condition using a periodic forcing yC, yWt. The " +
				"provided values of yWt, yC must be callable functions f(t), " +
				"rather than scalar values. ") from e
			# Set mapping (x; xp, up) -> (yWt(x), yC(x)) for initial condition
			self.advection_map = lambda x, xp, up: \
				advection_map(x, xp, up, (yWt_fn, yC_fn))
		self.f = plugin_steady_state.SteadyState(x_global, p_vent, inlet_input_val,
			input_type=input_type, # "u", "j", "p" as specified input
			override_properties=props,
			use_static_cache=True, # Optimize across processes by sharing global soln
		)
		# Save argument values
		self.x_global = np.expand_dims(x_global,axis=(1,2))
		self.p_vent = p_vent
		self.inlet_input = (inlet_input_val, input_type)

	def get_state(self, physics, x, t):
		# Evaluate intial condition with constant yC, yWt input
		U_init = self.f(x)

		if self.approx_massfracs:
			# Extract velocity from global x
			u = physics.compute_variable("XVelocity", self.f(self.x_global))
			# Construct approximate yWt(x), yC(x) with oscillatory input
			yWt, yC = self.advection_map(x, self.x_global, u)
			# Apply correction to water, crystal content
			rho = U_init[..., physics.get_mass_slice()].sum(axis=-1, keepdims=True)
			U_init[..., physics.get_state_slice("pDensityWt")] = yWt * rho
			U_init[..., physics.get_state_slice("pDensityC")]  = yC * rho
			# Remove lambdas for compatibility with pickle module
			self.advection_map = None

		return U_init

class UniformExsolutionTest(FcnBase):
	'''
	Uniform n-dimensional box for testing exsolution.
	'''

	def __init__(self, arhoA=0.0, arhoWv=0.8, arhoM=2500.0, u=0., T=1000., #arhoM=2496.3
		arhoWt=2500.0*0.04, arhoC=100.0):
		self.arhoA = arhoA
		self.arhoWv = arhoWv
		self.arhoM = arhoM
		self.u = u
		self.T = T
		self.arhoWt = arhoWt
		self.arhoC = arhoC

	def get_state(self, physics, x, t):
		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()
		arhoA = self.arhoA
		arhoWv = self.arhoWv
		arhoM = self.arhoM
		u = self.u
		T = self.T
		arhoWt = self.arhoWt
		arhoC = self.arhoC

		rho = arhoA+arhoWv+arhoM
		e = (arhoA * physics.Gas[0]["c_v"] * T + 
			arhoWv * physics.Gas[1]["c_v"] * T + 
			arhoM * (physics.Liquid["c_m"] * T + physics.Liquid["E_m0"])
			+ 0.5 * rho * u**2.)
		
		Uq[:, :, iarhoA] = arhoA
		Uq[:, :, iarhoWv] = arhoWv
		Uq[:, :, iarhoM] = arhoM
		Uq[:, :, imom] = rho * u
		Uq[:, :, ie] = e
		# Tracer quantities
		Uq[:, :, iarhoWt] = arhoWt
		Uq[:, :, iarhoC] = arhoC

		return Uq # [ne, nq, ns]


class IsothermalAtmosphere(FcnBase):
	'''
	Isothermal air atmosphere as an initial condition.
	'''

	def __init__(self,T:float=300., p_atm:float=1e5,
		h0:float=-150.0, hmax:float=6000.0, gravity:float=9.8,
		massFracWv:float=5e-3, massFracM:float=1e-7, tracer_frac:float=1e-7):
		''' Set atmosphere temperature, pressure, and location of pressure.
		Pressure distribution is computed as hydrostatic profile with p = p_atm
		at elevation h0.
		'''
		self.T = T
		self.p_atm = p_atm
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
		eval_pts = np.unique(x[:,:,1:2])
		# Evaluate IVP solution
		soln = scipy.integrate.solve_ivp(
			lambda height, p:
				-self.gravity / atomics.mixture_spec_vol(self.y, p, self.T, physics),
			[self.h0, self.hmax],
			[1e5],
			t_eval=eval_pts,
			dense_output=True)
		# Cache the pressure interpolant
		self.pressure_interpolant = soln.sol
		# Evaluate p using dense_output of solve_ivp (with shape magic)
		p = np.reshape(self.pressure_interpolant(x[...,1].ravel()), x[...,1].shape)
		rho = atomics.mixture_density(self.y, p, self.T, physics)
		phi = atomics.gas_volfrac(np.einsum("ij, k -> ijk", rho, self.y),
			self.T, physics)[:,:,0]
		# Compute partial densities
		arhoA = (phi * nA) * p / (physics.Gas[0]["R"] * self.T)
		arhoWv = (phi * nWv) * p / (physics.Gas[1]["R"] * self.T)
		arhoM = (1.0 - phi) * physics.Liquid["rho0"] \
			* (1.0 + (p - physics.Liquid["p0"])/physics.Liquid["K"])

		# Compute conservative variables for tracers (typically no source in 2D)
		arhoWt = arhoWv + self.tracer_frac * arhoM
		arhoC = self.tracer_frac * arhoM
		arhoFm = (1.0 - self.tracer_frac) * arhoM
		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)
		# Compute volumetric energy
		rho = arhoA + arhoWv + arhoM
		e = (arhoA * physics.Gas[0]["c_v"] * self.T + 
			arhoWv * physics.Gas[1]["c_v"] * self.T + 
			arhoM * (physics.Liquid["c_m"] * self.T + physics.Liquid["E_m0"])
			+ 0.5 * rho * u**2.)
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

class LinearAtmosphere(FcnBase):
	'''
	Constant-density atmosphere (isopycnic) as an initial condition. The
	Temperature is adjusted to keep the density constant (and thus the pressure
	a linear function of elevation).
	'''

	def __init__(self,T0:float=300., p_atm:float=1e5,
		h0:float=-150.0, gravity:float=9.8, massFracWv=5e-3, arhoMR=1e-9):
		''' Set atmosphere temperature, pressure, and location of pressure.
		Pressure distribution is computed as hydrostatic profile with p = p_atm
		at elevation h0.
		'''
		self.T0 = T0
		self.p_atm = p_atm
		self.h0 = h0
		self.gravity = gravity
		self.massFracWv = massFracWv
		self.arhoMR = arhoMR

	def get_state(self, physics, x, t):
		# Unpack
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		# Mass-weighted gas constant R (approx. yM ~ 0)
		R = (1.0 - self.massFracWv) * physics.Gas[0]["R"] \
			+ self.massFracWv * physics.Gas[1]["R"]
		# Compute scale height at reference temperature T0
		hs0 = R*self.T0/self.gravity
		# Compute pressure linear in elevation
		p = self.p_atm * (1.0 - (x[:,:,1:2] - self.h0)/hs0).squeeze(axis=2)
		# Compute approx. volume fraction correcting for water partial pressure
		prod = physics.Gas[0]["R"] * (1.0 - self.massFracWv)
		alphaA = prod / (prod + physics.Gas[1]["R"] * self.massFracWv)
		# Constant pure air density at h0
		arhoA = alphaA * self.p_atm / (physics.Gas[0]["R"]*self.T0)
		# Compute temperature
		T = alphaA * p / (arhoA * physics.Gas[0]["R"])
		# Zero or trace amounts of Wv, M and tracers
		arhoWv = (1.0 - alphaA) * p / (physics.Gas[1]["R"] * T)
		arhoM = self.arhoMR*np.ones_like(p)
		arhoWt = arhoWv
		arhoC = 0.1*self.arhoMR*np.ones_like(p) # In principle should be passive in 2D
		arhoFm = 0.9*self.arhoMR*np.ones_like(p)
		# Zero velocity
		u = np.zeros_like(p)
		v = np.zeros_like(p)

		rho = arhoA + arhoWv + arhoM

		e = (arhoA * physics.Gas[0]["c_v"] * T + 
			arhoWv * physics.Gas[1]["c_v"] * T + 
			arhoM * (physics.Liquid["c_m"] * T + physics.Liquid["E_m0"])
			+ 0.5 * rho * u**2.)
		
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


class NohProblem(FcnBase):
	'''
	Noh problem for axisymmetric testing in the (r,z) view. This is a shock
	propagation test in the r-direction only (to test the implementation of the
	radial geometric source term.)
	See doi:10.1115/1.4041195
	'''

	def __init__(self, eps=1e-3, rho0=1.0, u0=1.0):
		''' Set epsilon for pressure of the unshocked fluid, the density scale rho0,
		and the velocity scale u0 > 0 of the converging unshocked fluid (magnitude
		only).
		'''
		self.eps = eps
		self.rho0 = rho0
		self.u0 = u0

	def get_state(self, physics, x, t):
		# Initialize and get useful index names
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()
		# Take gamma of air
		gamma = physics.Gas[0]["gamma"]

		# Set pressure to epsilon (strong shock limit takes eps -> 0)
		p = self.eps
		# Compute energy resulting from epsilon
		e0 = 0.5 * self.rho0 * self.u0**2.0 + p / (gamma - 1)

		# Map single-phase (air) problem to multiphase state variables
		Uq[:, :, iarhoA]  = self.rho0
		Uq[:, :, iarhoWv] = 0.0
		Uq[:, :, iarhoM]  = 0.0
		Uq[:, :, irhou]   = -self.rho0 * self.u0
		Uq[:, :, irhov]   = 0.0
		Uq[:, :, ie]      = e0
		# Leave tracer quantities zero

		return Uq # [ne, nq, ns]


class MultipleRiemann(FcnBase):
	'''
	Replicates the Riemann IC in the section of the 1D domain far from the
	interface. Used as an initial condition for low-hassle Riemann problem
	tests. 
	'''
	def __init__(self, rhoL=1.0, uL=0.0, pL=1.0,
										 rhoR=0.125, uR=0.0, pR=0.1):
		self.rhoL = rhoL
		self.uL = uL
		self.pL = pL
		self.rhoR = rhoR
		self.uR = uR
		self.pR = pR

	def get_state(self, physics, x, t):
		Uq = np.zeros([x.shape[0], x.shape[1], physics.NUM_STATE_VARS])
		gamma = physics.gamma
		Rg = physics.R

		Uq[:, :, 0] = self.rhoL
		Uq[:, :, 1] = self.rhoL * self.uL
		Uq[:, :, 2] = self.pL / (gamma - 1.0) + \
											 0.5 * self.rhoL* np.power(self.uL, 2.0)
		
		Uq[np.logical_and(x[:,:,0] < -8, x[:,:,0] > -14), 0] = self.rhoR
		Uq[np.logical_and(x[:,:,0] < -8, x[:,:,0] > -14), 1] = self.rhoR * self.uR
		Uq[np.logical_and(x[:,:,0] < -8, x[:,:,0] > -14), 2] = self.pR / (gamma - 1.0) + \
											 0.5 * self.rhoR* np.power(self.uR, 2.0)

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

class SlipWall(BCWeakPrescribed):
	'''
	This class corresponds to a slip wall. See documentation for more
	details.
	'''
	def get_boundary_state(self, physics, UqI, normals, x, t):
		smom = physics.get_momentum_slice()

		# Unit normals
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)

		# Remove momentum contribution in normal direction from boundary
		# state
		rhoveln = np.sum(UqI[:, :, smom] * n_hat, axis=2, keepdims=True)
		UqB = UqI.copy()
		UqB[:, :, smom] -= rhoveln * n_hat

		return UqB
		
		# Ideal gas HACK: TODO: correction for
		if np.any(UqI[...,1] + UqI[...,2] > 0):
			raise NotImplementedError("Remove wall correction for non-ideal gas in functions.py")
		gamma = physics.Gas[0]["gamma"]

		# HACK to turn off enforcement on horizontal walls
		# if np.abs(n_hat[0,0,0]) < np.abs(n_hat[0,0,1]):
			# return UqB

		if False:
			# Compute isentropic deceleration quantity
			rho = UqI[:, :, 0]
			p = physics.compute_variable("Pressure", UqI)
			Z = 0.5 * (gamma - 1) * rhoveln.squeeze(axis=2)**2.0 / rho / p.squeeze(axis=2)

			# Compute density factor (density_new == densityFactor * density_old)
			# Exponent 1/(gamma - 1) applies for specific energy conservation and
			# 1/gamma applies for volumetric energy conservation
			densityFactor = (1 + Z) ** (1/(gamma - 1))
			# Prevent infinite compression for strong shock limit
			densityFactor = np.clip(densityFactor, None, 2e2)
			# Compute isentropically compressed mass density
			UqB[:, :, 0] *= densityFactor
			# Compute volumetric energy due to compressed mass density
			UqB[:, :, 5] *= densityFactor
		else:
			''' Rankine-Hugoniot boundary construction '''
			# Compute using inward normal
			rhoveln *= -1
			rhoveln = np.squeeze(rhoveln, axis=-1)
			# Get nonzero sign of velocity
			signum = np.sign(rhoveln)
			signum[rhoveln == 0] = -1
			# Define shock speed function
			shock_speed = lambda rho, u, p, e: (3-gamma)/4*u - signum * np.sqrt(
				((gamma-3)/4 * u)**2 + (gamma-1) * (e+p)/rho
			)
			# Compute density from air only
			rho = UqI[:,:,0]
			u = rhoveln / rho
			e = UqI[:,:,5]
			# Compute internal energy, removing all components
			e_int = e - 0.5 * np.linalg.norm(UqI[:,:,3:5], axis=-1)**2 / rho
			p = (gamma - 1) * e_int
			c = np.sqrt(gamma*p/rho)
			# Compute 1D projected shock speed
			# S = shock_speed(rho, u, p, e_int + 0.5*rho*u*u)
			S = shock_speed(rho, u, p, e)
			# Compute and set target states
			hat_rho = rho - rhoveln / S
			hat_p = p + rhoveln * (u - S)
			hat_e = e - u / S * (e + p)
			hat_u = 0
			UqB[:,:,0] = hat_rho
			UqB[:,:,5] = hat_e

		return UqB


class PressureOutlet(BCWeakPrescribed):
	'''
	This class corresponds to an outflow boundary condition with static
	pressure prescribed.
	The boundary state is computed by connecting the initial state to the state
	with the prescribed pressure along the integral curve of the extended
	eigenvector corresponding to the entering waves.

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, p):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			p: pressure

		Outputs:
		--------
				self: attributes initialized
		'''
		self.p = p

	def get_boundary_state(self, physics, UqI, normals, x, t):
		if physics.NDIMS > 1:
			raise NotImplementedError("Pressure boundary condition (multiphasevpT) only done for 1D")

		''' Perform physical checks '''
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		rhoI = UqI[:, :, physics.get_mass_slice()].sum(axis=2, keepdims=True)
		# Interior velocity in normal-tangential coordinates
		velI = UqI[:, :, physics.get_momentum_slice()]/rhoI
		velnI = np.sum(velI*n_hat, axis=2, keepdims=True)
		veltI = velI - velnI*n_hat
		pI = physics.compute_variable("Pressure", UqI)
		if np.any(pI < 0.):
			raise errors.NotPhysicalError
		# if np.any(velnI < 0.):
			# print("Incoming flow at outlet")

		''' Short-circuit function for sonic exit '''
		cI = physics.compute_variable("SoundSpeed", UqI)
		Mn = velnI/cI
		if np.any(Mn >= 1.):
			# If supersonic, then extrapolate interior to exterior
			return UqB

		UqB = UqI.copy()
		p_target = self.p

		# Define numerics for target state construction
		damping_idx_scale = 5
		additional_damping_factor = 1.0
		N_steps_max = 20
		# Permissive step count
		N_steps_max = np.max([N_steps_max, 10*damping_idx_scale])
		p_rel_tol = 1e-4

		# Initialize monitor variables for measuring relative change
		p = 1e-15
		p_last = 1e15
		stuck_counter = 0

		def compute_eig_aco_negative(U, physics):
			''' Computes eigenvector corresponding to u-c'''

			'''Compute rows of flux Jacobian for tracer states'''
			# Row of flux Jacobian for tracer states
			rho = U[:,:,physics.get_mass_slice()].sum(axis=2)
			u = U[:,:,physics.get_state_index("XMomentum")] / rho
			# Truncated row of flux Jacobian for tracer states
			#   b = [-u, -u, -u, 1, 0] / rho * q_i
			# where q_i is the partial density of the given tracer state
			N_states_hyperbolic = 5
			N_states_tracer = 2
			# Fill temporary construction vector with size [ne, nq, ns_hyp]
			b_sub = np.tile(np.zeros_like(u), (1,1,N_states_hyperbolic))
			b_sub[:,:,physics.get_mass_slice()] = -u/rho
			b_sub[:,:,physics.get_state_index("XMomentum")] = 1.0/rho
			# Fill temporary construction vector with size [ne, nq, ns_tracer]
			slice_like_tracers = (physics.get_state_index("pDensityWt"), 
				physics.get_state_index("pDensityC"))
			arho_tracers = U[:,:,slice_like_tracers]
			# Compute rows of flux Jacobian for tracer states
			b = np.einsum("ijk, ijl -> ijkl", arho_tracers, b_sub)

			''' Compute u-c eigenvector of hyperbolic subsystem '''
			# Size [ne, nq, ns_hyp]
			#   x = y1 y2 y3 u-a H - au
			# Mass fractions
			y1 = U[:,:,0] / rho
			y2 = U[:,:,1] / rho
			y3 = U[:,:,2] / rho
			eigvec_hyp = np.zeros_like(b_sub)
			H = physics.compute_additional_variable("TotalEnthalpy", U, True)
			a = physics.compute_additional_variable("SoundSpeed", U, True)
			eigvec_hyp[:,:,0] = y1
			eigvec_hyp[:,:,1] = y2
			eigvec_hyp[:,:,2] = y3
			eigvec_hyp[:,:,3] = u - a
			eigvec_hyp[:,:,4] = H - a*u

			''' Compute extension of the hyperbolic subsystem acoustic eigenvector'''
			# Compute (b^T * eigvec) / (eigval - u) -- 
			eigvec_ext = np.einsum("ijkl, ijl -> ijk", b, eigvec_hyp) / (-a)
			return np.concatenate((eigvec_hyp, eigvec_ext), axis=2)

		p_fn = lambda U : physics.compute_additional_variable("Pressure", U, True)
		dpdU_fn = lambda U : physics.compute_pressure_sgradient(U)
		f = lambda U: compute_eig_aco_negative(U, physics)

		for i in range(N_steps_max):
			''' Set damped Newton step size '''
			# Discrete Gaussian damping
			damping = 1.0 - np.exp(-((i+1)/damping_idx_scale)**2)
			damping *= additional_damping_factor
			# Compute Newton step size
			newton_step_size = (p_target - p_fn(UqB)) / \
					np.einsum("ijk, ijk -> ij", dpdU_fn(UqB), f(UqB))
			
			# Check for stalling
			# if (p - p_last) / p_target < p_rel_tol and i > 2 * damping_idx_scale:
			#     # Increase damping (high damping for monotonic approach to target)
			#     # (Deterministic alternative to stochastic perturbation)
			#     damping *= (0.9)**(stuck_counter+1)
			#     stuck_counter += 1
			p_last = p

			# Butcher table for Cash-Karp RK quadrature
			B = np.array([[1/5, 0, 0, 0, 0],
					[3/40, 9/40, 0, 0, 0],
					[3/10, -9/10, 6/5, 0, 0],
					[-11/54, 5/2, -70/27, 35/27, 0],
					[1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]])
			w = np.array([37/378, 0, 250/621, 125/594 , 0, 512/1771])
			# Compute damped step size for ODE integration
			damped_step_size = damping * newton_step_size
			# RK step 0
			num_stages = B.shape[0] + 1
			k = np.zeros(tuple(np.append([num_stages], list(UqB.shape))))
			k[0,:,:,:] = f(UqB)
			for j in range(B.shape[0]):
					k[j+1,:,:,:]= f(UqB + damped_step_size*
							np.einsum("m, mijk -> ijk", B[j,0:j+1], k[0:j+1,:]))
			UqB += damped_step_size * np.einsum("i, ijkl -> jkl", w, k)
			p = p_fn(UqB)

			if np.abs(p - p_target) / p_target < p_rel_tol:
				break

			if i == N_steps_max-1:
				print('''Boundary state construction reached max num steps allowed.
				To be replaced by a logger.warning in PressureOutlet
				''')

		# Compute final state
		# p = p_fn(UqB)
		# rho = UqB[:,:,0:3].sum(axis=2)
		# y1 = UqB[:,:,0] / rho
		# y2 = UqB[:,:,1] / rho
		# y3 = UqB[:,:,2] / rho
		# rhoA = UqB[:,:,0]/physics.compute_additional_variable("volFracA", UqB, True)
		# rhoWv = UqB[:,:,1]/physics.compute_additional_variable("volFracWv", UqB, True)
		# T = physics.compute_additional_variable("Temperature", UqB, True)
		# S = y1 * physics.Gas[0]["c_v"] * np.log(p / rhoA**physics.Gas[0]["gamma"]) + \
		# 		y2 * physics.Gas[1]["c_v"] * np.log(p / rhoWv**physics.Gas[1]["gamma"]) + \
		# 		y3 * physics.Liquid["c_m"] * np.log(T)
		# a = physics.compute_additional_variable("SoundSpeed", UqB, True)
		# beta = physics.compute_additional_variable("beta", UqB, True)
		# velx = UqB[:,:,3] / rho
		# Mn = velx / a

		''' Post-computation pressure check '''
		if np.any(p < 0.):
			raise errors.NotPhysicalError

		return UqB


class Inlet(BCWeakPrescribed):
	'''
	This class corresponds to an inflow boundary condition.
	The boundary state is computed by connecting the prescribed state to the state
	with the prescribed pressure along the integral curve of the extended
	eigenvector corresponding to the entering waves.

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, aux=None):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			p: pressure

		Outputs:
		--------
				self: attributes initialized
		'''
		self.U_chamber = np.array([
			0,
			1.2*1.257556105882443e+00,
			1.2*2.400000011294172e+03,
			0,													# Replaced variable
			1.2*7.202297102593764e+09,
			1.2*4.562862152986715e+01, # Abundance of total water
			1.2*4.307867810200626e+01])

	def get_boundary_state(self, physics, UqI, normals, x, t):
		if physics.NDIMS > 1:
			raise NotImplementedError("Pressure boundary condition (multiphasevpT) only done for 1D")

		''' Perform physical checks '''
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		rhoI = UqI[:, :, physics.get_mass_slice()].sum(axis=2, keepdims=True)
		# Interior velocity in normal-tangential coordinates
		velI = UqI[:, :, physics.get_momentum_slice()]/rhoI
		velnI = np.sum(velI*n_hat, axis=2, keepdims=True)
		veltI = velI - velnI*n_hat
		pI = physics.compute_variable("Pressure", UqI)
		if np.any(pI < 0.):
			raise errors.NotPhysicalError
		# if np.any(velnI < 0.):
		# 	print("Incoming flow at outlet")

		''' Short-circuit function for sonic exit '''
		cI = physics.compute_variable("SoundSpeed", UqI)
		Mn = velnI/cI
		if np.any(Mn >= 1.):
			# If supersonic, then extrapolate interior to exterior
			return UqB

		# Set initial to magma chamber values
		UqB = UqI.copy()
		UqB[:,:,:] = self.U_chamber

		# Define numerics for target state construction
		damping_idx_scale = 3
		additional_damping_factor = 1.0
		N_steps_max = 20
		# Permissive step count
		N_steps_max = np.max([N_steps_max, 10*damping_idx_scale])
		p_rel_tol = 1e-4

		# Initialize monitor variables for measuring relative change
		p = 1e-15
		p_last = 1e15
		stuck_counter = 0

		def compute_eig_aco_negative(U, physics):
			''' Computes eigenvector corresponding to u-c'''

			'''Compute rows of flux Jacobian for tracer states'''
			# Row of flux Jacobian for tracer states
			rho = U[:,:,physics.get_mass_slice()].sum(axis=2)
			u = U[:,:,physics.get_state_index("XMomentum")] / rho
			# Truncated row of flux Jacobian for tracer states
			#   b = [-u, -u, -u, 1, 0] / rho * q_i
			# where q_i is the partial density of the given tracer state
			N_states_hyperbolic = 5
			N_states_tracer = 2
			# Fill temporary construction vector with size [ne, nq, ns_hyp]
			b_sub = np.tile(np.zeros_like(u), (1,1,N_states_hyperbolic))
			b_sub[:,:,physics.get_mass_slice()] = -u/rho
			b_sub[:,:,physics.get_state_index("XMomentum")] = 1.0/rho
			# Fill temporary construction vector with size [ne, nq, ns_tracer]
			slice_like_tracers = (physics.get_state_index("pDensityWt"), 
				physics.get_state_index("pDensityC"))
			arho_tracers = U[:,:,slice_like_tracers]
			# Compute rows of flux Jacobian for tracer states
			b = np.einsum("ijk, ijl -> ijkl", arho_tracers, b_sub)

			''' Compute u-c eigenvector of hyperbolic subsystem '''
			# Size [ne, nq, ns_hyp]
			#   x = y1 y2 y3 u-a H - au
			# Mass fractions
			y1 = U[:,:,0] / rho
			y2 = U[:,:,1] / rho
			y3 = U[:,:,2] / rho
			eigvec_hyp = np.zeros_like(b_sub)
			H = physics.compute_additional_variable("TotalEnthalpy", U, True)
			a = physics.compute_additional_variable("SoundSpeed", U, True)
			eigvec_hyp[:,:,0] = y1
			eigvec_hyp[:,:,1] = y2
			eigvec_hyp[:,:,2] = y3
			eigvec_hyp[:,:,3] = u - a
			eigvec_hyp[:,:,4] = H - a*u

			''' Compute extension of the hyperbolic subsystem acoustic eigenvector'''
			# Compute (b^T * eigvec) / (eigval - u) -- 
			eigvec_ext = np.einsum("ijkl, ijl -> ijk", b, eigvec_hyp) / (-a)
			return np.concatenate((eigvec_hyp, eigvec_ext), axis=2)

		p_fn = lambda U : physics.compute_additional_variable("Pressure", U, True)
		dpdU_fn = lambda U : physics.compute_pressure_sgradient(U)
		f = lambda U: compute_eig_aco_negative(U, physics)

		for i in range(N_steps_max):
			''' Set damped Newton step size '''
			# Discrete Gaussian damping
			damping = 1.0 - np.exp(-((i+1)/damping_idx_scale)**2)
			damping *= additional_damping_factor

			# Check sonic orthogonality (eigenvector orth. to objective gradient, i.e.,
			# is sonic). Dot prod is r_i * grad_q objective = r_i * grad_q (rho*u)
			dot_prod = f(UqB)[:,:,3]
			if np.abs(dot_prod.squeeze()) < 1e-10:
				# TODO: logger.info
				break
			# Compute momentum Newton step size
			newton_step_size = (UqI[:,:,3] - UqB[:,:,3]) / dot_prod
			
			# Check for stalling
			# if (p - p_last) / p_target < p_rel_tol and i > 2 * damping_idx_scale:
			#     # Increase damping (high damping for monotonic approach to target)
			#     # (Deterministic alternative to stochastic perturbation)
			#     damping *= (0.9)**(stuck_counter+1)
			#     stuck_counter += 1
			p_last = p

			# Butcher table for Cash-Karp RK quadrature
			B = np.array([[1/5, 0, 0, 0, 0],
					[3/40, 9/40, 0, 0, 0],
					[3/10, -9/10, 6/5, 0, 0],
					[-11/54, 5/2, -70/27, 35/27, 0],
					[1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]])
			w = np.array([37/378, 0, 250/621, 125/594 , 0, 512/1771])
			# Compute damped step size for ODE integration
			damped_step_size = damping * newton_step_size
			# RK step 0
			num_stages = B.shape[0] + 1
			k = np.zeros(tuple(np.append([num_stages], list(UqB.shape))))
			k[0,:,:,:] = f(UqB)
			for j in range(B.shape[0]):
					k[j+1,:,:,:]= f(UqB + damped_step_size*
							np.einsum("m, mijk -> ijk", B[j,0:j+1], k[0:j+1,:]))
			UqB += damped_step_size * np.einsum("i, ijkl -> jkl", w, k)

			# Objective evaluation
			p = UqB[:,:,3]
			p_target = UqI[:,:,3]

			if np.isclose(p, p_target) or np.abs(p - p_target) / p_target < p_rel_tol:
				break

			if i == N_steps_max-1:
				print('''Boundary state construction reached max num steps allowed.
				To be replaced by a logger.warning in PressureOutlet
				''')

		# Compute final state
		# p = p_fn(UqB)
		# rho = UqB[:,:,0:3].sum(axis=2)
		# y1 = UqB[:,:,0] / rho
		# y2 = UqB[:,:,1] / rho
		# y3 = UqB[:,:,2] / rho
		# rhoA = UqB[:,:,0]/physics.compute_additional_variable("volFracA", UqB, True)
		# rhoWv = UqB[:,:,1]/physics.compute_additional_variable("volFracWv", UqB, True)
		# T = physics.compute_additional_variable("Temperature", UqB, True)
		# S = y1 * physics.Gas[0]["c_v"] * np.log(p / rhoA**physics.Gas[0]["gamma"]) + \
		# 		y2 * physics.Gas[1]["c_v"] * np.log(p / rhoWv**physics.Gas[1]["gamma"]) + \
		# 		y3 * physics.Liquid["c_m"] * np.log(T)
		# a = physics.compute_additional_variable("SoundSpeed", UqB, True)
		# beta = physics.compute_additional_variable("beta", UqB, True)
		# velx = UqB[:,:,3] / rho
		# Mn = velx / a

		''' Post-computation pressure check '''
		if np.any(p < 0.):
			raise errors.NotPhysicalError

		return UqB


class CustomInlet(BCWeakPrescribed):
	'''
	This class corresponds to an inflow boundary condition that specifies mass
	and momentum flux over the boundary.

	Requires physics class to have a vent_physics(VentPhysics) object.

	Attributes:
	-----------
	None
	'''
	def __init__(self):
		self.porousLength = 1.0
		self.poreLengthScale = 1e-3
		self.porousDragCoeff = 1e-6 # Function of Re, porosity, pore length scale
		self.porousDragMultiplier = self.porousDragCoeff * self.porousLength \
																/ self.poreLengthScale

	def resolveReverseFlow (self, pGrid, pPocket, JPlus, cOut, gamma):
		''' Computes boundary values of pressure and M for subsonic flow
				from atmosphere to pocket.
		
		Args:
			pGrid (np.array): DG domain boundary pressure
			pPocket: Pocket pressure
			JPlus: DG domain J+ == u + 2*c / (gamma-1)
			cOut: DG domain soudn speed
			gamma: Heat capacity ratio
			frictionMultiplier: C_d * L / bar{d} for porous flow

		Returns:
			(p, M): Pressure, Mach number tuple
		'''

		# Flag to use scalar-array type conversions and debug output
		use_safe_mode = False

		globalPressureRatio = pGrid / pPocket

		# (Subsonic) Mach number as a function of (p/p_pocket)^2
		MFn = lambda P2 : JPlus / cOut * np.power(globalPressureRatio / np.sqrt(P2), (gamma-1)/(2*gamma)) - 2 / (gamma - 1)
		# Fixed point mapping for variable (p/p_pocket)^2 representing
		# the equation P2 == 1 / (1 - 2*(...) * (MFn(P2))^2 )
		fixedPointMapping = lambda P2 : 1.0 / (1.0 - 2.0 * gamma * self.porousDragMultiplier
			* np.power(MFn(P2),2.0))

		''' Perform fixed point iteration '''
		# Set initial guess 
		pRatioLastSq = np.ones(pGrid.shape)
		pRatioSq = fixedPointMapping(pRatioLastSq)
		iterCount = 1
		while np.any(np.abs(pRatioSq - pRatioLastSq)/ pRatioLastSq > 1e-5) and iterCount < 10:
			pRatioLastSq = pRatioSq
			pRatioSq = fixedPointMapping(pRatioSq)
			iterCount += 1
		
		pRatio = np.sqrt(pRatioSq)
		M = JPlus / cOut * np.power(globalPressureRatio / pRatio,
																(gamma-1)/(2*gamma)) - 2 / (gamma - 1)

		if use_safe_mode:
			print(f"Debug: Iteration count: {iterCount}")
			# Scalar to array cast, or no-op
			M = np.array(M)
			pRatio = np.array(pRatio)

		''' Check choked condition '''
		pRatioChoked =  np.sqrt(1.0 + 2.0*gamma*self.porousDragMultiplier)

		M[pRatio >= pRatioChoked] = 1.0 / pRatioChoked
		pRatio[pRatio >= pRatioChoked] = pRatioChoked
		
		# Compute pressure from pRatio
		p = pRatio * pPocket
		return (p, M)

	def get_boundary_state(self, physics, UqI, normals, x, t, aux_output=None):

		# Init
		UqB = UqI.copy()
		# Unpack
		srho = physics.get_state_slice("Density")
		srhoE = physics.get_state_slice("Energy")
		smom = physics.get_momentum_slice()
		gamma = physics.gamma
		# Compute unit normals
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)

		''' Get interior state '''
		# Extrapolate interior state to boundary
		interior = physics.vent_physics.createVentLocalState()
		interior.rho = UqI[:, :, srho]
		interior.vel = UqI[:, :, smom] / interior.rho
		interior.veln = np.sum(interior.vel*n_hat[:, :, :],
													 axis=2,
													 keepdims=True)
		interior.p = physics.compute_variable("Pressure", UqI)[:, :, :]
		if np.any(interior.p < 0.0):
			raise errors.NotPhysicalError
		interior.c = physics.compute_variable("SoundSpeed", UqI)[:, :, :]
		interior.Mn = interior.veln / interior.c
		# Interior tangential velocity
		interior.velt = interior.vel - interior.veln*n_hat[:, :, :]
		interior.x = x[:,:,:]

		''' Get boundary data profile '''
		# Uniform exit profile
		inletProfileVelocity = 1.0 + 0.0*x[:,:,0:1]
		inletProfilePressure = inletProfileVelocity
		inletProfileSoundSpeed = inletProfileVelocity
		# # Smooth exit profile
		# inletProfileVelocity = np.expand_dims(
		# 	np.exp(1.0 - 1.0 / (1.0 - np.power(x,2.0))),
		# 	axis=2)
		# smoootherCutoffLength = 1.0
		# inletProfileVelocity[np.all(abs(x[ventMask,:,0]) >=
		#                    smoootherCutoffLength,axis=1), :, :] = 0.0

		''' Get pocket state '''
		pocket = physics.vent_physics.createVentLocalState()
		pocket.rho = physics.vent_physics.rho
		pocket.p = physics.vent_physics.p
		pocket.c = np.sqrt(physics.vent_physics.gamma * physics.vent_physics.p
											/ physics.vent_physics.rho)

		''' Get partial porous exit state '''
		boundary = physics.vent_physics.createVentLocalState()
		boundary.c = pocket.c * inletProfileSoundSpeed
		boundary.velt = interior.velt
		boundary.x = x[:,:,:]

		# # Evaluate boundary exterior state as a function of x
		# pExt = physics.vent_physics.p * ventProfile
		# rhoExt = physics.vent_physics.rho * ventProfile
		# vnExt = 1e-6*np.sqrt(np.clip(pExt - pI,0,None))		# Porous law
		# cExt = np.sqrt(gamma * pExt / rhoExt)

		''' Identify grid-sonic points '''
		gridSonicMask = np.all(interior.Mn <= -1.0,axis=1).squeeze()

		''' Compute subsonic inflow hypothetical '''
		# Compute porous exit condition based on interior invariant JOut
		JOut = interior.veln + 2.0*interior.c/(gamma - 1.)
		boundary.veln = interior.veln + 2.0/(gamma - 1.0) \
											* (interior.c - boundary.c)
		# Compute compressible, steady-state, isothermal porous flow state
		boundary.Mn = boundary.veln / boundary.c
		boundary.p = pocket.p * inletProfilePressure / np.sqrt(
			1 + 2.0 * gamma * self.porousDragMultiplier * np.power(boundary.Mn, 2.0)
		)
		boundary.rho = gamma * boundary.p / np.power(pocket.c, 2.0)

		# Compute invariants at boundary
		J0 = boundary.p / np.power(boundary.rho, gamma)
		# assert(np.all(boundary.veln <= 0)) # TEST
		JIn = boundary.veln - 2.0*boundary.c/(gamma - 1.)

		# Boundary state
		# boundary = CustomInlet.LocalState()
		# boundary.c = 0.25*(gamma - 1.0)*(JOut - JIn)
		# boundary.velt = interior.velt
		# boundary.veln = 0.5*(JOut + JIn)
		# boundary.Mn = boundary.veln / boundary.c
		# boundary.rho = np.power(np.power(boundary.c, 2.0) / (gamma * J0), 1.0/(gamma - 1.0))
		# boundary.p = boundary.rho * np.power(boundary.c, 2.0) / gamma

		''' Identify boundary-sonic points '''
		boundarySonicMask = np.all(boundary.Mn <= -1.0,axis=1).squeeze()

		''' Union of sonic points '''
		sonicMask = np.logical_or.reduce([gridSonicMask,
															 boundarySonicMask])

		''' Recompute at sonic points '''
		boundary.veln[sonicMask, :, :] = -boundary.c[sonicMask, :, :]
		boundary.Mn[sonicMask, :, :] = -1.0
		boundary.p[sonicMask, :, :] = pocket.p * \
			inletProfilePressure[sonicMask, :, :] / np.sqrt(
				1.0 + 2.0 * gamma * self.porousDragMultiplier * np.power(-1.0, 2.0)
		)
		boundary.rho[sonicMask, :, :] = gamma * boundary.p[sonicMask, :, :] \
																			/ np.power(boundary.c[sonicMask, :, :], 2.0)
		# JOut[sonicMask, :, :] = boundary.veln[sonicMask, :, :] \
														# + 2.0 / (gamma - 1.0) * boundary.c[sonicMask, :, :]		

		''' Compute outflow points
				Outflow points are masked based on grid-value M and boundary M.
		'''

		# rhoveln = np.sum(UqI[:, :, smom] * n_hat[:, :, :],
		#                  axis=2, keepdims=True)
		outflowMask = np.all(np.logical_or(interior.Mn > 0.0, boundary.Mn > 0.0),axis=1).squeeze()
		# Linearizing pressure in terms of quantity (M^2)
		linearizedPressureDrop = pocket.p * (1.0
			 + gamma * self.porousDragMultiplier * np.power(boundary.Mn, 2.0))

		pSubs, MSubs = self.resolveReverseFlow (
			interior.p[outflowMask, :, :], 
			pocket.p, 
			JOut[outflowMask, :, :],
			interior.c[outflowMask, :, :],
			gamma)

		boundary.p[outflowMask, :, :] = pSubs
		boundary.Mn[outflowMask, :, :] = MSubs
		boundary.c[outflowMask, :, :] = JOut[outflowMask, :, :] \
																		/(boundary.Mn[outflowMask, :, :] + 2.0 / (gamma - 1))
		boundary.rho[outflowMask, :, :] = gamma * boundary.p[outflowMask, :, :] \
																			/ np.power(boundary.c[outflowMask, :, :], 2.0)
		boundary.veln[outflowMask, :, :] = boundary.Mn[outflowMask, :, :] \
																			 * boundary.c[outflowMask, :, :]

		qqqqqqq = 1

		''' Obsolete: set pressure outflow '''
		# # Set equal pressures (neglecting porous flow pressure drop--may be a bad idea)
		# # boundary.p[outflowMask, :, :]	= pocket.p
		# boundary.rho[outflowMask, :, :] = interior.rho[outflowMask, :, :] * np.power(
		# 	pocket.p / interior.p[outflowMask, :, :], 1.0 / gamma
		# )
		# boundary.veln[outflowMask, :, :] = interior.veln[outflowMask, :, :] \
		# 	+ 2.0/(gamma - 1.0) * (interior.c[outflowMask, :, :] - np.sqrt(
		# 		gamma * boundary.p[outflowMask, :, :] / boundary.rho[outflowMask, :, :]
		# 	))

		''' Finalize boundary state '''		
		boundary.vel = boundary.veln*n_hat[:, :, :] + boundary.velt

		# Boundary kinetic energy (per volume)
		kineticB = 0.5*boundary.rho*np.sum(np.power(boundary.vel, 2.0),
																			 axis=2, keepdims=True)
		# Set boundary states
		UqB[:, :, srho] = boundary.rho
		UqB[:, :, smom] = boundary.rho * boundary.vel
		UqB[:, :, srhoE] = boundary.p / (gamma - 1.) + kineticB

		''' Export intermediate states '''
		if aux_output is not None:
			aux_output["pocket"] = pocket
			aux_output["interior"] = interior
			aux_output["boundary"] = boundary

		return UqB


class BCWeakLLF(BCBase):
	'''
	This class computes the boundary numerical local Lax-Friedrichs flux using the
	exterior state.
	'''

	def __init__(self):
		self.is_cross_dimension = False
		# Initalize logger
		# self.logger = logging.getLogger(f"{__name__}{hash(self)}")
		# self.logger.setLevel(logging.DEBUG)
		# h = logging.FileHandler(
		# 	filename=f"BCWeakLLF_{hash(self)}.log",
		# 	encoding="utf-8")
		# h.setFormatter(logging.Formatter(
		# 	'[%(asctime)s][%(levelname)s] Logger <%(name)s> : %(message)s'))
		# self.logger.addHandler(h)

	def get_boundary_flux(self, physics, UqI, normals, x, t, gUq=None):
		
		# F = physics.get_conv_flux_numerical(UqI, UqB, normals)
		# UqB = self.get_boundary_state(physics, UqI, normals, x, t)
		# F,_ = physics.get_conv_flux_projected(UqB, normals)
		
		if physics.NDIMS == 1 and self.is_cross_dimension:
			# Squish 2D state to 1D face-averaged state
			# UqB_down = np.mean(UqB,axis=(0,1),keepdims=True)[:,:,mask_omit_xmomentum]
			# F = physics.get_conv_flux_numerical(UqI, UqB_down, normals)
			# UqB = UqB_down

			UqB = self.get_boundary_state(
				physics, UqI, normals, x, t, get_average=True)		
			# Reduce 2D momentum to 1D momentum
			mask_omit_xmomentum = [i for i in np.array(range(UqB.shape[2]))
				if not i == physics.get_state_index("XMomentum")]
			UqB = UqB[:,:,mask_omit_xmomentum]
			F = physics.get_conv_flux_numerical(UqI, UqB, normals)
		else:
			UqB = self.get_boundary_state(physics, UqI, normals, x, t, get_average=False)
			F = physics.get_conv_flux_numerical(UqI, UqB, normals)

		log_state = False
		if log_state:
			# Initalize logger
			self.logger = logging.getLogger(f"{__name__}{hash(self)}")
			self.logger.setLevel(logging.DEBUG)
			h = logging.FileHandler(
				filename=f"BCWeakLLF_{hash(self)}.log",
				encoding="utf-8")
			h.setFormatter(logging.Formatter(
				'[%(asctime)s][%(levelname)s] Logger <%(name)s> : %(message)s'))
			# Hack: logging (conflicts when multiple proesses)
			if len(self.logger.handlers)== 0:
				self.logger.addHandler(h)

			self.logger.info(f"t = {t}")
			self.logger.info(f"UqI (interior) = {UqI}")
			self.logger.info(f"UqB (outboundary) = {UqB}")
			self.logger.info(f"F = {F}")

		# Compute diffusive boundary fluxes if needed
		if physics.diff_flux_fcn:
			raise NotImplementedError
			Fv, FvB = physics.get_diff_boundary_flux_numerical(UqI, UqB, 
					gUq, normals) # [nf, nq, ns]
			F -= Fv
			return F, FvB # [nf, nq, ns], [nf, nq, ns, ndims]
		else:
			return F, None

class CouplingBC(BCWeakLLF):
	'''
	This class corresponds to an coupled boundary that allows inflow or outflow.

	Attributes:
	-----------
	bkey (immutable): Key for the boundary relevant to the instantiated BC
	t: Last update time
	bstate (CouplingBC.LocalState): Local state data object
	'''

	# @dataclass
	class LocalState:
		''' Data namespace for states at a point in the boundary sequence'''
		U: np.array     = np.array([])   # State in native domain
		Ucast: np.array = np.array([])   # State U, casted to n-t coords
		Ulift: np.array = np.array([])   # State U, lifted to common dim
		vel: np.array   = np.array([])
		veln: np.array  = np.array([])
		velt: np.array  = np.array([])
		# Mn: np.array    = np.array([])
		# p: np.array     = np.array([])
		a: np.array     = np.array([])
		# rho: np.array   = np.array([])
		x: np.array     = np.array([])
		n_hat: np.array = np.array([])
		Ucross: np.array = np.array([])  # Cross-domain size

	# (Debug) Seconds to wait for bdry data before raising exception
	maxWaitTime = 5.0
	
	class NetworkTimeoutError(Exception):
		pass

	def __init__(self, bkey):
		super().__init__()
		self.bkey = bkey
		self.bstate = CouplingBC.LocalState()

	def get_extrapolated_state(self, physics, UqI, normals, x, t):
		''' Extrapolate interior state to boundary, expose LocalState object
				State is valid at time self.t, and can be async accessed when
				t == self.t.
		'''
		self.bstate = CouplingBC.LocalState()
		# Unpack
		# srho = physics.get_state_slice("Density")
		# srhoE = physics.get_state_slice("Energy")
		# smom = physics.get_momentum_slice()
		# Compute unit normals
		self.bstate.n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		self.bstate.U = UqI
		self.bstate.vel = UqI[:, :, physics.get_momentum_slice()] / \
			np.sum(UqI[:, :, physics.get_mass_slice()],axis=2,keepdims=True)
		self.bstate.veln = np.sum(self.bstate.vel*self.bstate.n_hat[:, :, :],
			axis=2, keepdims=True)
		# self.bstate.p = physics.compute_variable("Pressure", UqI)[:, :, :]
		self.bstate.a = physics.compute_variable("SoundSpeed", UqI)[:, :, :]
		# self.bstate.Mn = self.bstate.veln / self.bstate.c
		# Interior tangential velocity
		self.bstate.velt = self.bstate.vel - self.bstate.veln*self.bstate.n_hat[:, :, :]
		self.bstate.x = x[:,:,:]

		return self.bstate

	@abstractmethod
	def get_boundary_state(self, physics, UqI, normals, x, t):
		pass
		# ''' Called when computing states at shared boundaries. '''
		# raise NotImplementedError("Abstract CouplingBC class was instantiated. " +
		# 													" Implement get_boundary_state.")


class MultiphasevpT2D1D(CouplingBC):
	'''
	This class couples a 2D multiphase domain to a 1D multiphase domain.
	The representation of the solution at the boundary is done as follows:
		* Broadcasts 1D states to 2D

	Attributes:
	-----------
	None
	'''

	def __init__(self, bkey):
		super().__init__(bkey)
		self.bdry_flux_fcn = LaxFriedrichs2D()
		self.is_cross_dimension = True

	def get_extrapolated_state(self, physics, UqI, normals, x, t):
		''' Called when computing states at shared boundaries. '''

		super().get_extrapolated_state(physics, UqI, normals, x, t)
		self.bstate.Ucast = self.bstate.U.copy()
		
		if physics.NDIMS == 1:
			# Lift boundary state U to 2D
			self.bstate.Ulift = np.copy(self.bstate.U)
			# Insert XMomentum(2D), reassign XMomentum(1D) -> YMomentum(2D)
			self.bstate.Ulift = np.insert(self.bstate.Ulift,
				physics.get_momentum_slice(),
				self.bstate.Ulift[:,:,physics.get_momentum_slice()], axis=2)
			# Set XMomentum(2D) = 0
			self.bstate.Ulift[:,:,physics.get_momentum_slice()] = 0.0
			
			np.append(self.bstate.Ucast, 
								self.bstate.Ucast[:,:,-1:], axis=2)
			self.bstate.Ucast = np.append(self.bstate.Ucast, 
								self.bstate.Ucast[:,:,-1:], axis=2)
			# Set tangential momentum to zero
			self.bstate.Ucast[:,:,physics.get_momentum_slice()] = 0.0
			self.bstate.Ucross = self.bstate.Ulift.copy()
		elif physics.NDIMS == 2:
			# Project to [rho, ... , rho vel_n, rho vel_t, rho e, ...]
			rho = self.bstate.Ucast[:,:,physics.get_mass_slice()].sum(
				axis=2,keepdims=True)
			self.bstate.Ucast[:,:,physics.get_momentum_slice()] = \
				rho * self.bstate.veln
			# TODO: check consistency of implementation. Tangentials are required to
			# prevent self-amplification in the 2D domain
			# TODO: check floating point cancellation 
			rotation_cw = np.array([ [0, 1], [-1, 0], ])
			t_hat = np.einsum('mk, ijk', rotation_cw, self.bstate.n_hat)
			self.bstate.Ucast[:,:,physics.get_momentum_slice()] = np.sum(
				(self.bstate.vel - self.bstate.veln * self.bstate.n_hat) * t_hat,
				axis=2,
				keepdims=True)
			self.bstate.Ulift = self.bstate.U.copy()
			self.bstate.Ucross = self.bstate.U.copy()
		else:
			raise Exception("Unhandled NDIMS in get_extrapolated_state. 0D?")
		
		return self.bstate
		
	def compute_roe_flux(self, physics2D, UqSelf, UqAdj):
		''' Roe flux computation copied from class Roe1D, adapted to normal inputs '''
		# (see physics.conv_flux_fcn.compute_flux)
		# N.B. the coordinate system is pre-rotated to the normal coordinates
		# N.B. UqAdj has its own normal coordinates (here we take negative)

		# Unpack
		srho = physics2D.get_state_slice("Density")
		smom = physics2D.get_momentum_slice()
		gamma = physics2D.gamma

		# # Unit normals
		# n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)

		# Allocate memory for boundary flux function
		self.bdry_flux_fcn = Roe2D(UqSelf)
		# self.bdry_flux_fcn.R = 0.0*UqSelf
		# self.bdry_flux_fcn.alphas = 0.0*UqSelf
		# self.bdry_flux_fcn.alphas = 0.0*UqSelf

		# Get velL, velR -- essentially normal
		# Rebind Uq
		UqAdj = UqAdj.copy()
		UqSelf = UqSelf.copy()
		# Account for opposite unit normal and opposite tangent orientation
		UqAdj[:,:,smom] *= -1.0

		# UqAdj[:,:,2] = 0.0
		# UqSelf[:,:,2] = 0.0

		velL = UqSelf[:,:,smom] / UqSelf[:,:,srho]
		velR = UqAdj[:,:,smom] / UqAdj[:,:,srho]
		rhoRoe, velRoe, HRoe = self.bdry_flux_fcn.roe_average_state(physics2D,
														 srho, velL, velR, UqSelf, UqAdj)

		# Speed of sound from Roe-averaged state
		c2 = (gamma - 1.)*(HRoe - 0.5*np.sum(velRoe*velRoe, axis=2,
				keepdims=True))
		if np.any(c2 <= 0.):
			# Non-physical state
			raise errors.NotPhysicalError
		c = np.sqrt(c2)

		# Jumps
		drho, dvel, dp = 	self.bdry_flux_fcn.get_differences(
			physics2D, srho, velL, velR, UqSelf, UqAdj)
		# alphas (left eigenvectors multiplied by dU)
		alphas = 	self.bdry_flux_fcn.get_alphas(c, c2, dp, dvel, drho, rhoRoe)
		# Eigenvalues
		evals = 	self.bdry_flux_fcn.get_eigenvalues(velRoe, c)
		# Right eigenvector matrix
		R = self.bdry_flux_fcn.get_right_eigenvectors(
			c, 
			self.bdry_flux_fcn.get_eigenvalues(velRoe, c),
			velRoe,
			HRoe
		)
		# Form flux Jacobian matrix multiplied by dU
		dURoe = np.einsum('ijkl, ijl -> ijk', R, np.sign(evals)*alphas)

		return .5*(UqSelf + UqAdj - dURoe) # [nf, nq, ns]

	def get_boundary_state(self, physics, UqI, normals, x, t, get_average=False):
		adjacent_domain_id = [
			key for key in 
			physics.domain_edges[physics.domain_id][self.bkey] 
			if key != physics.domain_id][0]
		data_net_key = physics.edge_to_key(
										 physics.domain_edges[physics.domain_id][self.bkey],
										 adjacent_domain_id)
		# Get shared state
		# Flat manager-dict model
		adjacent_bstate = copy.deepcopy(physics.bdry_data_net[data_net_key])
		# Nested manager-dict model
		# adjacent_bstate = copy.deepcopy(
		# 	physics.bdry_data_net[physics.domain_id][self.bkey] \
		# 	[adjacent_domain_id]["payload"])

		#
		if get_average:
			return adjacent_bstate["bdry_face_state_averaged"]
		else:
			return adjacent_bstate["bdry_face_state"].Ulift

		# Prep useful parameters
		gamma = physics.gamma
		n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)

		# DEPRECATE:
		adjacent_physics_NDIMS = 3 - physics.NDIMS

		if physics.NDIMS < adjacent_physics_NDIMS: # from 1D side
			# TODO: replace with proper integration
			adjacentUCast = np.mean(adjacent_bstate.Ucast, axis=(0,1), keepdims=True)
			# Remove tangential and return
			adjacentUCast = np.concatenate((adjacentUCast[:,:,0:2], adjacentUCast[:,:,3:4]), axis=2)
			# print(physics.compute_variable("Pressure", adjacentUCast))
			# Convert from n-t representation to x-y
			# Correct for opposite orientation
			adjacentUCast[:,:,1] *= -1.0
			return adjacentUCast
			return np.mean(adjacent_bstate.Ucast, axis=(0,1), keepdims=True)[0,1]
		elif physics.NDIMS > adjacent_physics_NDIMS: # from 2D side
			adjacentUCast = np.tile(adjacent_bstate.Ucast, (self.bstate.Ucast.shape[0],self.bstate.Ucast.shape[1],1))
			# adjacentUCast[:,:,1] *= 1.0

			# Convert from n-t representation to x-y
			# TODO: save t_hat as state in network payload
			rotation_cw = np.array([ [0, 1], [-1, 0], ])
			t_hat = np.einsum('ijk, mk', self.bstate.n_hat, rotation_cw)	
			n_hat = -self.bstate.n_hat
			rhovel_n = adjacentUCast[:,:,1:2]
			rhovel_t = adjacentUCast[:,:,2:3]
			rhovel_n_repr = np.einsum('ijk, ijn -> ijk', n_hat, rhovel_n)
			rhovel_t_repr = np.einsum('ijk, ijn -> ijk', t_hat, rhovel_t)
			adjacentUCast[:,:,1:3] = rhovel_n_repr + rhovel_t_repr
			return adjacentUCast


		# Perform dimension matching
		dim_match = lambda data : data
		if physics.NDIMS < adjacent_physics_NDIMS:
			# TODO: replace with proper integration
			dim_match = lambda data : np.expand_dims(
				np.expand_dims(np.array([np.mean(data)]), axis=1), axis=2)
		elif physics.NDIMS > adjacent_physics_NDIMS:
			# Manual broadcast
			dim_match = lambda data : data.squeeze() * np.ones(self.bstate.c.shape)

		# Assume uniform 1D and broadcast to copy
		bcasted_shape = 0.0*(self.bstate.Ucast + adjacent_bstate.Ucast)
		selfUcast = bcasted_shape + self.bstate.Ucast
		adjacentUcast = bcasted_shape + adjacent_bstate.Ucast

		# TODO: Generalize to all variables, separate code for 2D
		# Opposite orientation at boundary
		if adjacent_bstate.x.size > 1 and x.size > 1:
			adjacentUcast = np.flip(adjacentUcast, axis=(0,1))
			assert(np.linalg.norm(np.flip(adjacent_bstate.x, axis=(0,1)) - x) < 1e-4)
		else: # 2D1D
			# TODO: Should we delete tangentials?
			# selfUcast[:,:,2:3] = 0.0
			# adjacentUcast[:,:,2:3] = 0.0
			pass

		return adjacentUcast

		# Compute Roe state as boundary state
		if physics.NDIMS == 2:
			URoe = self.compute_roe_flux(physics, selfUcast,
				adjacentUcast)
		else:
			from physics.euler.euler import Euler2D
			# Trick Euler2D constructor into thinking this is a real domain
			# TODO: use better workaround
			class Fake:
				pass
			fake_mesh = Fake()
			fake_mesh.boundary_groups = {}
			fake_mesh.ndims = 2
			fake_physics2D = Euler2D(fake_mesh)
			fake_physics2D.gamma = physics.gamma
			fake_physics2D.R = physics.R
			URoe = self.compute_roe_flux(fake_physics2D, selfUcast,
				adjacentUcast)
		
		def compute_bdry_case(Mn):
			# Compute out-supersonic case
			JOut = self.bstate.veln + 2.0 / (gamma - 1.0) * self.bstate.c
			J0 = self.bstate.p / np.power(self.bstate.rho, gamma)
			JIn = self.bstate.veln - 2.0 / (gamma - 1.0) * self.bstate.c

			# Replace invariants with corresponding exterior state
			np.putmask(
				JIn,
				Mn < 1,
				dim_match(
					(-adjacent_bstate.veln) - 2.0 / (gamma - 1.0) * adjacent_bstate.c
			))
			np.putmask(
				J0,
				Mn < 0,
				dim_match(
					adjacent_bstate.p / np.power(adjacent_bstate.rho, gamma)
			))
			np.putmask(
				JOut,
				Mn <= -1,
				dim_match(
					(-adjacent_bstate.veln) + 2.0 / (gamma - 1.0) * adjacent_bstate.c
			))

			# Compute states based on invariants
			veln = 0.5 * (JIn + JOut)
			c = 0.25 * (gamma - 1)  * (JOut - JIn)
			rho = np.power(np.power(c, 2.0) / (gamma * J0), 1.0/(gamma - 1.0))
			p = rho * np.power(c, 2.0) / gamma
			vel = veln * self.bstate.n_hat + self.bstate.velt

			return veln, c, rho, p, vel

		# (veln, c, rho, p, vel) =  compute_bdry_case(self.bstate.Mn)

		# print(f"Mn1={np.mean(veln/c)};")
		# # Update
		# (veln, c, rho, p, vel) =  compute_bdry_case(veln / c)
		# print(f"Mn2={np.mean(veln/c)};")

		
		if physics.NDIMS == 1:
			# Dimension down by deleting tangential velocity and taking mean
			# TODO: mean needs be replaced with integration
			UqB = np.mean(np.delete(URoe, 2, axis=2),axis=(0,1), keepdims=True)
			# Adjust for boundary sign
			UqB[:,:,1:2] *= n_hat
		else:
			# Rotate
			UqB = URoe.copy()
			# Setting zero tangential velocity may be needed for a well-posed
			# Euler-Euler coupling, else unstable in the tangential direction (?)
			# UqB[:,:,1] = URoe[:,:,1] * n_hat[:,:,0] + self.bstate.velt[:,:,0]
			# UqB[:,:,2] = URoe[:,:,1] * n_hat[:,:,1] + self.bstate.velt[:,:,1]
			rotation_cw = np.array([ [0, 1], [-1, 0], ])
			t_hat = np.einsum('ijk, mk', self.bstate.n_hat, rotation_cw)
			UqB[:,:,1] = URoe[:,:,1] * n_hat[:,:,0] + URoe[:,:,2] * t_hat[:,:,0]
			UqB[:,:,2] = URoe[:,:,1] * n_hat[:,:,1] + URoe[:,:,2] * t_hat[:,:,1]
		
		# Try y-filter
		# May be needed for well-posed coupling condition, else unstable in the tangential direction (?)
		# UqB[:,:,1] = 0

		# Set boundary states	
		
		# UqB = UqI.copy()
		# kineticB = 0.5*rho*np.sum(np.power(vel, 2.0),
		# 																	 axis=2, keepdims=True)
		# UqB[:, :, physics.get_state_slice("Density")] = rho
		# UqB[:, :, physics.get_momentum_slice()] = rho * vel
		# UqB[:, :, physics.get_state_slice("Energy")] = p / (gamma - 1.) + kineticB

		assert(not np.any(np.isnan(UqB)))
		assert(UqB.shape == UqI.shape)

		return UqB


class MultiphasevpT2D1DCylindrical():
	''' TODO: inherit MultiphasevpT2D1D, replacing the boundary integration of
	int (f) dx with the radially weighted int (f * 2r/a) dr, where a is the
	conduit radius. This is equivalent to replacing int (f) dx / int dx with
	int (f) 2 pi r dr / int 2 pi r dr. The factor of 2 in 2r/a can be seen from
	computing the average value of the function f = 1, which involves integration
	of r -> r^2/2. '''
	pass


class MultiphasevpT2D2DCylindrical():
	pass


class Euler2D2D(BCWeakRiemann):
	'''
	This class couples a 2D Euler domain to a 2D Euler domain.
	Implementation mimics the Roe approximate RIemann solve for interior fluxes.

	Attributes:
	-----------
	bkey (str): Key corresponding to boundary as edge in domain graph; typically
							the name of the boundary.
	'''

	def __init__(self, bkey):
		self.bkey = bkey

	def get_extrapolated_state(self, physics, UqI, normals, x, t):
		''' Called when computing states at shared boundaries. '''
		return UqI

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Get shared state UqI and return (BCWeakRiemann) '''
		# Get id string of adjacent domain from domain graph (local)
		adjacent_domain_id = [
			key for key in 
			physics.domain_edges[physics.domain_id][self.bkey] 
			if key != physics.domain_id][0]
		# Get key for boundary state in shared memory (via Manager.dict)
		data_net_key = physics.edge_to_key(
										 physics.domain_edges[physics.domain_id][self.bkey],
										 adjacent_domain_id)
		# Return orientation-corrected exterior state
		return np.flip(
			copy.copy(physics.bdry_data_net[data_net_key]),
			axis=(0,1))
		

class MultiphasevpT1D1D(BCWeakRiemann):
	'''
	This class implements a 1D-1D coupling boundary condition.
	TODO: The numerical flux for the artifical viscosity diffusion integral
	need to be dealt with.

	Attributes:
	-----------
	bkey (str): Key corresponding to boundary as edge in domain graph; typically
							the name of the boundary.
	'''

	def __init__(self, bkey):
		self.bkey = bkey

	def get_extrapolated_state(self, physics, UqI, normals, x, t):
		''' Called when computing states at shared boundaries. '''
		return UqI

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Get shared state UqI and return (BCWeakRiemann) '''
		# Get id string of adjacent domain from domain graph (local)
		adjacent_domain_id = [
			key for key in 
			physics.domain_edges[physics.domain_id][self.bkey] 
			if key != physics.domain_id][0]
		# Get key for boundary state in shared memory (via Manager.dict)
		data_net_key = physics.edge_to_key(
										 physics.domain_edges[physics.domain_id][self.bkey],
										 adjacent_domain_id)
		# Return orientation-corrected exterior state
		return np.flip(
			copy.copy(physics.bdry_data_net[data_net_key]["bdry_face_state"]),
			axis=(0,1))


class MultiphasevpT2D2D(BCWeakRiemann):
	'''
	This class implements a 2D-2D coupling boundary condition.
	TODO: The numerical flux for the artifical viscosity diffusion integral
	need to be dealt with.

	Attributes:
	-----------
	bkey (str): Key corresponding to boundary as edge in domain graph; typically
							the name of the boundary.
	'''

	def __init__(self, bkey):
		self.bkey = bkey

	def get_extrapolated_state(self, physics, UqI, normals, x, t):
		''' Called when computing states at shared boundaries. '''
		return UqI

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Get shared state UqI and return (BCWeakRiemann) '''
		# Get id string of adjacent domain from domain graph (local)
		adjacent_domain_id = [
			key for key in 
			physics.domain_edges[physics.domain_id][self.bkey] 
			if key != physics.domain_id][0]
		# Get key for boundary state in shared memory (via Manager.dict)
		data_net_key = physics.edge_to_key(
										 physics.domain_edges[physics.domain_id][self.bkey],
										 adjacent_domain_id)
		# Return orientation-corrected exterior state
		return np.flip(
			copy.copy(physics.bdry_data_net[data_net_key]["bdry_face_state"]),
			axis=(0,1))


class NonReflective1D(BCWeakPrescribed):
	'''
	This class corresponds to a nonreflective outflow boundary condition.
	The boundary state is computed using acoustic Riemann invariants in 1D.
	Work in progress: linearized radiation/impedance boundary condition.

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, p):
		self.p = p
		self.initialized = False

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		UqB = UqI.copy()

		# Short-circuiting
		# return UqB

		''' Compute normal velocity. '''
		# n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		# rhoI = UqI[:, :, physics.get_mass_slice()].sum(axis=2, keepdims=True)
		arhoVec = UqI[:,:,physics.get_mass_slice()]
		rhoI = atomics.rho(arhoVec)
		
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
		uGrid = physics.compute_variable("XVelocity", UqI)
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
		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho_target * yI
		UqB[:,:,physics.get_momentum_slice()] = rho_target * uHat
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho_target * yI, physics) * THat \
			+ (rho_target * yI[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * rho_target * uHat * uHat
		''' Update adiabatically compressed/expanded tracer partial densities '''
		UqB[:,:,5:] *= rho_target / rhoI

		''' Post-computation validity check '''
		if np.any(THat < 0.):
			raise errors.NotPhysicalError

		return UqB


class LinearizedIsothermalOutflow2D(BCWeakPrescribed):
	'''
	Provides an outflow 

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self):
		# Instantiate isothermal atmosphere (not bound to initial condition)
		# self.isothermal_atmosphere = IsothermalAtmosphere()
		self.initialized = False
		self.U0 = None

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		# Fetch isothermal
		if not self.initialized:
			self.initialized = True

			# Fetch isothermal atmosphere IC
			init_func = physics.IC
			# Check type of initial condition
			if type(physics.IC) is not physics.IC_fcn_map.get(
					FcnType.IsothermalAtmosphere, None):
				raise ValueError("LinearizedIsothermalOutflow2D used but initial " +
					"condition set by IsothermalAtmosphere was not found.")

			# Fill pressure from the initial condition
			p0 = x[...,1:].copy()
			p0.ravel()[:] = init_func.pressure_interpolant(x[...,1:].ravel())
			self.p0 = p0
			# Unpack temperature
			T = init_func.T
			# Compute intermediate states
			rho = atomics.mixture_density(init_func.y, p0, T, physics)
			phi = atomics.gas_volfrac(np.einsum("ij, k -> ijk",
				rho[...,0], init_func.y), T, physics)
			# Compute partial densities
			arhoA = (phi * init_func.nA) * p0 / (physics.Gas[0]["R"] * T)
			arhoWv = (phi * init_func.nWv) * p0 / (physics.Gas[1]["R"] * T)
			arhoM = (1.0 - phi) * physics.Liquid["rho0"] \
				* (1.0 + (p0 - physics.Liquid["p0"])/physics.Liquid["K"])
			# Compute conservative variables for tracers (typically no source in 2D)
			arhoWt = arhoWv + init_func.tracer_frac * arhoM
			arhoC = init_func.tracer_frac * arhoM
			arhoFm = (1.0 - init_func.tracer_frac) * arhoM
			# Get initial state for linearization
			self.U0 = np.zeros((*x.shape[:-1], physics.NUM_STATE_VARS))
			self.U0[...,0:1] = arhoA
			self.U0[...,1:2] = arhoWv
			self.U0[...,2:3] = arhoM
			self.U0[...,5:6] = atomics.c_v(self.U0[...,0:3], physics) * T
			self.U0[...,6:7] = arhoWt
			self.U0[...,7:8] = arhoC
			self.U0[...,8:9] = arhoFm

			self.Z0 = physics.compute_variable("SoundSpeed", self.U0) * rho

		''' Compute mass variables '''
		arhoVec = UqI[:,:,physics.get_mass_slice()]
		rhoI = atomics.rho(arhoVec)
		yVec = UqI[:,:,physics.get_mass_slice()] / rhoI

		''' Compute normal velocity '''
		n_hat = normals / np.linalg.norm(normals, axis=2, keepdims=True)
		rhoveln = (UqI[:, :, physics.get_momentum_slice()] * n_hat).sum(
			axis=2, keepdims=True)
		veln = rhoveln / rhoI

		''' Compute interior pressure '''
		TI = atomics.temperature(arhoVec, UqI[:, :, physics.get_momentum_slice()],
			UqI[:, :, physics.get_state_slice("Energy")], physics)
		pI = atomics.pressure(
			arhoVec, TI, atomics.gas_volfrac(arhoVec, TI, physics), physics)
		Gamma = atomics.Gamma(arhoVec, physics)
		
		''' Compute linearized characteristic quantities in units of velocity '''
		p0 = self.p0
		char_out = (pI - p0)/self.Z0 + veln
		char_in = (pI - p0)/self.Z0 - veln
		velnB = 0.5 * char_out
		pB = p0 + 0.5 * self.Z0 * char_out
		# Assess ingoing characteristic quantity due to nonlinearity
		pass

		''' Evaluate state '''
		# Compute isentropic temperature change
		TB = TI * (pB / pI)**((Gamma-1.0)/Gamma)
		rhoB = atomics.mixture_density(yVec, pB, TB, physics)
		# Map to kinematic variables
		UqB = UqI.copy()
		UqB /= rhoI
		# Change face-normal velocity component
		UqB[:, :, physics.get_momentum_slice()] += (velnB - veln) * n_hat
		# Map to conservative variables
		UqB *= rhoB
		# Fill energy explicitly
		kinetic_energy = 0.5 * np.expand_dims(np.einsum("...i, ...i -> ...",
			UqB[:, :, physics.get_momentum_slice()],
			UqB[:, :, physics.get_momentum_slice()]), axis=-1) / rhoB
		UqB[:, :, physics.get_state_slice("Energy")] = \
			atomics.c_v(arhoVec, physics) * TB + kinetic_energy

		if np.any((pB < 0.0) | (TB < 0.0)):
			raise errors.NotPhysicalError

		return UqB


class PressureOutlet1D(BCWeakRiemann):
	'''
	This class corresponds to an outflow boundary condition with static
	pressure prescribed.
	The boundary state is computed using acoustic Riemann invariants in 1D.
	Since the Riemann invariants are constants along the eigenvector path, it is
	equivalent to integrate along the eigenvector corresponding to outgoing waves;
	mixture entropy per mass is held constant; pressure and density are
	related by the sound speed; pressure and velocity are related by the acoustic
	impedance. To capture choked flow efficiently, the approach here is to compute
	velocity in the following initial value problem (IVP) in state space:
		du/dp = 1/(rho*c),
		u(p = p0) = u0
	where rho*c is a function of pressure at constant entropy. The boundary state
	is computed as a function of self.p and u(p = self.p) when the flow is
	subsonic. Otherwise, sonic pressure is found, and the boundary state is
	computed as a function of p_sonic and u(p = p_sonic).
	The choking condition is treated as a termination event for the wave speed
	u - c, as a function of p. Note that rho, c are functions of p, and this
	initial value problem form is sufficiently general.

	Attributes:
	-----------
	p: float
		pressure
	'''
	def __init__(self, p):
		self.p = p

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		UqB = UqI.copy()

		''' Compute 1D velocity. '''
		# n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		arhoVec = UqI[:,:,physics.get_mass_slice()]
		rhoI = atomics.rho(arhoVec)
		velI = UqI[:, :, physics.get_momentum_slice()]/rhoI

		''' Compute interior pressure. '''
		# pI = physics.compute_variable("Pressure", UqI)		
		TI = atomics.temperature(arhoVec,
			UqI[:,:,physics.get_momentum_slice()], 
			UqI[:,:,physics.get_state_slice("Energy")], physics)
		gas_volfrac = atomics.gas_volfrac(arhoVec, TI, physics)
		pI = atomics.pressure(arhoVec, TI, gas_volfrac, physics)
		if np.any(pI < 0.):
			raise errors.NotPhysicalError
		# Compute mass-weighted, conserved quantities
		Gamma = atomics.Gamma(arhoVec, physics)
		yI = atomics.massfrac(arhoVec)

		''' Handle sonic outflow or inflow based on interior. '''
		cI = atomics.sound_speed(Gamma, pI, rhoI, gas_volfrac, physics)
		if np.any(velI >= cI):
			return UqB
		elif np.any(velI < 0):
			raise ValueError("Inflow. Check fragmentation state (front too close?) " +
				"and stability with respect to timestepper.")

		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to ingoing acoustic waves) of the interior
		solution. '''
		# Reformat shape of IVP initial conditions in state space
		T1, p1, u1 = TI.ravel()[0], pI.ravel()[0], np.array([velI.ravel()[0],])
		# Set target pressure (unchanged if flow is indeed subsonic)
		p_target = self.p
		# Define affine mapping from progress variable t in [0,1] to pressure p in
		#   [min(p1, p2), max(p1, p2)]
		t2p = lambda t: p1 + t*(p_target-p1)
		# Precompute fluid composition constants
		am2 = physics.Liquid["K"] / physics.Liquid["rho0"]
		yR_g = yI[...,0].squeeze() * physics.Gas[0]["R"] \
			+ yI[...,1].squeeze() * physics.Gas[1]["R"]
		yM = yI[...,2].squeeze()
		p_intercept = physics.Liquid["p0"] - physics.Liquid["K"]

		def Y_s(p):
			''' Efficient implementation of the isentropic admittance function. '''
			rhom_am2 = p - p_intercept
			return np.sqrt((
				yR_g * T1 * (p/p1)**((Gamma-1.0)/Gamma) / (p*p)
				+ yM * am2 / (rhom_am2*rhom_am2)
			) / Gamma)
		
		class ChokingEvent():
			''' Initial value problem termination event, returning u - c. '''
			def __init__(self):
				self.terminal = True

			def __call__(self, t, u):
				# Map progress variable t to pressure p
				p = t2p(t)
				# Compute temperature along isentrope
				T = T1 * (p / p1) **((Gamma-1.0)/Gamma)
				# Compute mixture density
				rho = atomics.mixture_density(yI, p, T, physics)
				# Return u - c
				return u - atomics.sound_speed(Gamma, p, rho,
					atomics.gas_volfrac(rho*yI, TI, physics), physics).ravel()[0]
		
		# Solve IVP for progress variable on [0,1], checking for choking, and 
		#   defaulting to one-step if possible
		soln_obj = scipy.integrate.solve_ivp(
			lambda t, u: -normals * (p_target-p1) * Y_s(t2p(t)),
			(0, 1), u1, vectorized=True, first_step=1.0, events=[ChokingEvent()])
		# Unpack solution dependent on flow choking
		if len(soln_obj.y_events[0]) == 0:
			# Extract velocity at target pressure
			u_target = soln_obj.y.ravel()[-1]
		else:
			# Extract velocity at choke
			u_target = soln_obj.y_events[0][0]
			# Re-evaluate target pressure as sonic pressure
			p_target = t2p(soln_obj.t_events[0][0])
		# Compute temperature along isentrope
		T_target = T1 * (p_target / p1) **((Gamma-1.0)/Gamma)

		''' Map computed (u_target, p_target, T_target) to conservative variables '''
		rho_target = atomics.mixture_density(yI, p_target, T_target, physics)
		UqB[:,:,physics.get_mass_slice()] = rho_target * yI
		UqB[:,:,physics.get_momentum_slice()] = rho_target * u_target
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho_target * yI, physics) * T_target \
			+ (rho_target * yI[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * rho_target * u_target * u_target
		''' Update adiabatically compressed/expanded tracer partial densities '''
		UqB[:,:,5:] *= rho_target / rhoI

		''' Post-computation validity check '''
		if np.any(T_target < 0.) or np.any(p_target < 0.):
			raise errors.NotPhysicalError
		
		return UqB


class PressureOutlet2D(BCWeakPrescribed):
	pass


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


class PressureStableLinearizedInlet1D(BCWeakPrescribed):
	'''
	Linearized inlet conditions that solve a linearized version of the Riemann
	problem with uL == uR. The Riemann problem consists of a jump in the state
	variables at the boundary between the designated (inlet) values and the
	numerical trace from the interior of the numerical domain. The constraints
	for the problem are:
		1. Entropy and mass fractions are given by the inlet values
		2. Particle velocity is continuous (uL == uR)
		3. Pressure of the inlet is used in the Riemann problem (pL) but is not
		     necessarily the pressure used at the boundary.
	The entering shock is linearized as an acoustic (expansion) wave, resulting
	in a three-wave structure.

	Asymptotically, the pressure at the boundary becomes that of the inlet (pL),
	motivating the name "PressureStable".
	Also, by construction, the entropy and mass fractions at the boundary are
	those of the inlet.
	There is no asymptotic statement about the velocity at the boundary, since
	this in general depends on the interior solution.

	The problem linearized on both sides of the contact wave has solution
	  p = (YL * pL + YR * pR) / (YL + YR) + (uL - uR) / (YL + YR)
	and the assumption of continuous particle velocity drops the latter term.

	Usage:
	  cVFav sets the averages crystal volume fraction.
	  For a constant state boundary, set is_gaussian=False, cos_freq=0.0.
	  For a sinusoidal boundary, set is_gaussian=False, cos_freq > 0. The
		  amplitude is controlled by cVFamp, with average cVFav.
		For a gaussian boundary, set is_gaussian=True. cos_freq is ignored.
		  The gaussian is controlled by gaussian_tpulse, gaussian_sig, and
		  cVFamp, with background value cVFav.
	Water content is specified as a concentration per dry magma mass (chi_water).
	'''
	def __init__(self, p_chamber:float=100e6, T_chamber:float=1e3, trace_arho:float=1e-6,
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
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		gas_volfrac = atomics.gas_volfrac(arhoVecI, TGrid, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid, gas_volfrac, physics)
		rhoGrid = atomics.rho(arhoVecI)
		GammaGrid = atomics.Gamma(arhoVecI, physics)
		cGrid = atomics.sound_speed(GammaGrid, pGrid, rhoGrid, gas_volfrac, physics)
		uGrid = momxI / rhoGrid
		# Compute chamber mass properties
		arhoVecChamber = arhoVecI.copy()
		arhoVecChamber[...,0] = self.trace_arho # Small amount of air to preserve positivity
		arhoVecChamber[...,1] = self.trace_arho # Small amount of water to preserve positivity
		# Approximate partial density of magma by density
		arhoVecChamber[...,2] = rho0 * (1 + (p_chamber - p0) / K)
		rhoChamber = atomics.rho(arhoVecChamber)
		GammaChamber = atomics.Gamma(arhoVecChamber, physics)
		cChamber = atomics.sound_speed(GammaChamber, p_chamber, rhoChamber,
			atomics.gas_volfrac(arhoVecChamber, T_chamber, physics), physics)

		# Compute composition-based properties
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# Outflow
			Gamma = GammaGrid
			y = atomics.massfrac(arhoVecI)
			S = TGrid / pGrid**((Gamma-1)/Gamma)
		else:
			# Inflow
			Gamma = GammaChamber
			y = atomics.massfrac(arhoVecChamber)
			S = T_chamber / p_chamber**((Gamma-1)/Gamma)

		# Compute acoustic impedances Z and admittances Y = 1/Z
		ZGrid = rhoGrid * cGrid
		ZChamber = rhoChamber * cChamber
		YGrid = 1.0 / ZGrid
		YChamber = 1.0 / ZChamber
		# Approximate pressure as Y-weighted average
		p = (YGrid * pGrid + YChamber * p_chamber) / (YGrid + YChamber)
		# Approximate corresponding velocity in x-direction TODO: generalize with face normal
		u = uGrid + (p_chamber - pGrid) / (ZChamber + ZGrid)

		# Evaluate temperature from isentropic outgoing wave
		T = S * p**((Gamma-1)/Gamma)
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

class VelocityInlet1D(BCWeakPrescribed):
	'''
	Blah blah blah TODO: copy from MassFluxInlet1D
	'''
	def __init__(self, u:float=1.0, p_chamber:float=100e6,
			T_chamber:float=1e3, trace_arho:float=1e-6, freq:float=0.25, # 1/36
			yWt:float=0.04, yC:float=0.1,
			use_linearized:bool=True, newton_tol:float=1e-7, newton_iter_max=20):
		# Ingest args
		self.u, self.p_chamber, self.T_chamber, self.trace_arho = \
			u, p_chamber, T_chamber, trace_arho
		# Angular frequency of variation
		self.freq = freq
		self.yWt, self.yC, self.use_linearized, self.newton_tol, self.newton_iter_max = \
			yWt, yC, use_linearized, newton_tol, newton_iter_max

	def get_linearized_boundary_state(self, physics, UqI, normals, x, t):
		''' Compute a boundary state by replacing Riemann problem with acoustic
		waves, and then approximating the acoustic wave. '''
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

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
		# Approximate desired mass flux
		u = self.u
		
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		gas_volfrac = atomics.gas_volfrac(arhoVecI, TGrid, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid, gas_volfrac, physics)
		rhoGrid = atomics.rho(arhoVecI)
		GammaGrid = atomics.Gamma(arhoVecI, physics)
		cGrid = atomics.sound_speed(GammaGrid, pGrid, rhoGrid, gas_volfrac, physics)
		uGrid = momxI / rhoGrid
		# Compute composition-based properties
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# Outflow
			Gamma = atomics.Gamma(arhoVecI, physics)
			y = atomics.massfrac(arhoVecI)
			S = TGrid / pGrid**((Gamma-1)/Gamma)
		else:
			# Inflow
			arhoVecB = arhoVecI.copy()
			arhoVecB[...,0] = self.trace_arho # Small amount of air to preserve positivity
			arhoVecB[...,1] = self.trace_arho # Small amount of water to preserve positivity
			# Approximate partial density of magma by density
			arhoVecB[...,2] = rho0 * (1 + (p_chamber - p0) / K)
			Gamma = atomics.Gamma(arhoVecB, physics)
			y = atomics.massfrac(arhoVecB)
			S = T_chamber / p_chamber**((Gamma-1)/Gamma)

		# Specific gas constant for R
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Evaluate primitive variables for boundary state
		p = pGrid - (rhoGrid * cGrid) * (uGrid - u)
		T = S * p**((Gamma-1)/Gamma)
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

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		if self.use_linearized:
			# Delegate to linearized
			return self.get_linearized_boundary_state(physics, UqI, normals, x, t)
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		# n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# TODO: improve out flow and sonic handling
			print("Attempting to outflow into an inlet")
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to outgoing acoustic waves) of the interior
		solution. '''
		# Extract data from input and prescribed chamber/reservoir values
		arhoVecI = UqI[:,:,physics.get_mass_slice()]
		momxI = UqI[...,physics.get_momentum_slice()]
		eI = UqI[...,physics.get_state_slice("Energy")]
		j, p_chamber, T_chamber = self.mass_flux, self.p_chamber, self.T_chamber
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		# Compute intermediates
		Gamma = atomics.Gamma(arhoVecI, physics)
		y = atomics.massfrac(arhoVecI)
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Compute chamber entropy as ratio T/p^(..)
		S_r = T_chamber / p_chamber**((Gamma-1)/Gamma)
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid,
			atomics.gas_volfrac(arhoVecI, TGrid, physics), physics)

		def eval_fun_dfun(p):
			''' Evaluate function G and its derivative dG/dp. '''
			# Define reusable groups
			g1 = yRGas * p**(-1/Gamma) * S_r
			g2 = y[...,2] * K / (rho0*(p + K - p0))
			# Integration of 1/impedance
			# Note that the output p, T are not used, since entropy is not taken from grid
			_, uTarget, _ = atomics.velocity_RI_fixed_p_quadrature(p, UqI, physics, normals,
				is_adaptive=True, tol=1e-1, rtol=1e-5)
			# Evaluate integrand
			f = atomics.acousticRI_integrand_scalar(np.array(p), TGrid, pGrid, y, Gamma, physics)
			# Evaluate returns
			G = j * (g1 + g2) + normals * uTarget
			dGdp = -j * (g1 * (1/Gamma) / p + g2 / (p+K-p0)) - f
			return G, dGdp, (p, uTarget)

		# Perform Newton iteration to compute boundary p
		p = pGrid.copy()
		for i in range(self.newton_iter_max):
			G, dGdp, _ = eval_fun_dfun(p)
			p -= G / dGdp
			if np.abs(G) < self.newton_tol:
				break
		# TODO: set logging if max iter is reached
		# Evaluate primitive variables for boundary state
		_, _, (p, u) = eval_fun_dfun(p)
		T = S_r * p**((Gamma-1)/Gamma)
		rho = atomics.mixture_density(y, p, T, physics)

		''' Check positivity of computed state. '''
		if np.any(T < 0.) or np.any(p < 0) or np.any(rho < 0):
			raise errors.NotPhysicalError

		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho * y
		UqB[:,:,physics.get_momentum_slice()] = j
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho * y, physics) * T \
			+ (rho * y[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * j * u
		# Update adiabatically compressed/expanded tracer partial densities
		UqB[:,:,5:] *= rho / atomics.rho(arhoVecI)
		#phi_crys = 0.4025 * (1.1 - 0.1 * np.cos(2 * np.pi * 0.25 * t)) # crystal vol / suspension vol
		#chi_water = 0.05055
		#UqB[:,:,5] = rho * (1 - phi_crys) / (1 + chi_water)
		#UqB[:,:,6] = rho * phi_crys

		return UqB


class VelocityInlet1D_neutralSinusoid(BCWeakPrescribed):
	'''
	sinusoidal crystal content injection where average remains constant (mostly copied from VelocityInlet1D)
	Blah blah blah TODO: copy from MassFluxInlet1D
	'''
	def __init__(self, u:float=1.0, p_chamber:float=100e6,
			T_chamber:float=1e3, trace_arho:float=1e-6, freq:float=0.25, # 1/36
			yWt:float=0.04, yC:float=0.1, cVFav:float=0.4,
			use_linearized:bool=True, newton_tol:float=1e-7, newton_iter_max=20):
		# Ingest args
		self.u, self.p_chamber, self.T_chamber, self.trace_arho = \
			u, p_chamber, T_chamber, trace_arho
		# Angular frequency of variation
		self.freq = freq
		self.cVFav = cVFav # average crystal volume fraction
		self.yWt, self.yC, self.use_linearized, self.newton_tol, self.newton_iter_max = \
			yWt, yC, use_linearized, newton_tol, newton_iter_max

	def get_linearized_boundary_state(self, physics, UqI, normals, x, t):
		''' Compute a boundary state by replacing Riemann problem with acoustic
		waves, and then approximating the acoustic wave. '''
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

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
		# Approximate desired mass flux
		u = self.u
		
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		gas_volfrac = atomics.gas_volfrac(arhoVecI, TGrid, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid, gas_volfrac, physics)
		rhoGrid = atomics.rho(arhoVecI)
		GammaGrid = atomics.Gamma(arhoVecI, physics)
		cGrid = atomics.sound_speed(GammaGrid, pGrid, rhoGrid, gas_volfrac, physics)
		uGrid = momxI / rhoGrid
		# Compute composition-based properties
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# Outflow
			Gamma = atomics.Gamma(arhoVecI, physics)
			y = atomics.massfrac(arhoVecI)
			S = TGrid / pGrid**((Gamma-1)/Gamma)
		else:
			# Inflow
			arhoVecB = arhoVecI.copy()
			arhoVecB[...,0] = self.trace_arho # Small amount of air to preserve positivity
			arhoVecB[...,1] = self.trace_arho # Small amount of water to preserve positivity
			# Approximate partial density of magma by density
			arhoVecB[...,2] = rho0 * (1 + (p_chamber - p0) / K)
			Gamma = atomics.Gamma(arhoVecB, physics)
			y = atomics.massfrac(arhoVecB)
			S = T_chamber / p_chamber**((Gamma-1)/Gamma)

		# Specific gas constant for R
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Evaluate primitive variables for boundary state
		p = pGrid - (rhoGrid * cGrid) * (uGrid - u)
		T = S * p**((Gamma-1)/Gamma)
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
		ta = 5
		tb = 5 + (1 / (2 * self.freq))
		if t < ta:
			phi_crys = self.cVFav
		elif t < tb:
			phi_crys = self.cVFav * (0.875 + 0.125 * np.cos(2 * np.pi * self.freq * (t - ta)))
		else:
			phi_crys = self.cVFav * (1 - 0.25 * np.cos(2 * np.pi * self.freq * (t - tb)))
		chi_water = 0.03
		UqB[:,:,5] = rho * chi_water * (1.0 - phi_crys) / (1 + chi_water)
		UqB[:,:,6] = rho * phi_crys
	
		# Fragmented state
		UqB[:,:,7] = 0.0

		return UqB

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		if self.use_linearized:
			# Delegate to linearized
			return self.get_linearized_boundary_state(physics, UqI, normals, x, t)
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		# n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# TODO: improve out flow and sonic handling
			print("Attempting to outflow into an inlet")
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to outgoing acoustic waves) of the interior
		solution. '''
		# Extract data from input and prescribed chamber/reservoir values
		arhoVecI = UqI[:,:,physics.get_mass_slice()]
		momxI = UqI[...,physics.get_momentum_slice()]
		eI = UqI[...,physics.get_state_slice("Energy")]
		j, p_chamber, T_chamber = self.mass_flux, self.p_chamber, self.T_chamber
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		# Compute intermediates
		Gamma = atomics.Gamma(arhoVecI, physics)
		y = atomics.massfrac(arhoVecI)
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Compute chamber entropy as ratio T/p^(..)
		S_r = T_chamber / p_chamber**((Gamma-1)/Gamma)
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid,
			atomics.gas_volfrac(arhoVecI, TGrid, physics), physics)

		def eval_fun_dfun(p):
			''' Evaluate function G and its derivative dG/dp. '''
			# Define reusable groups
			g1 = yRGas * p**(-1/Gamma) * S_r
			g2 = y[...,2] * K / (rho0*(p + K - p0))
			# Integration of 1/impedance
			# Note that the output p, T are not used, since entropy is not taken from grid
			_, uTarget, _ = atomics.velocity_RI_fixed_p_quadrature(p, UqI, physics, normals,
				is_adaptive=True, tol=1e-1, rtol=1e-5)
			# Evaluate integrand
			f = atomics.acousticRI_integrand_scalar(np.array(p), TGrid, pGrid, y, Gamma, physics)
			# Evaluate returns
			G = j * (g1 + g2) + normals * uTarget
			dGdp = -j * (g1 * (1/Gamma) / p + g2 / (p+K-p0)) - f
			return G, dGdp, (p, uTarget)

		# Perform Newton iteration to compute boundary p
		p = pGrid.copy()
		for i in range(self.newton_iter_max):
			G, dGdp, _ = eval_fun_dfun(p)
			p -= G / dGdp
			if np.abs(G) < self.newton_tol:
				break
		# TODO: set logging if max iter is reached
		# Evaluate primitive variables for boundary state
		_, _, (p, u) = eval_fun_dfun(p)
		T = S_r * p**((Gamma-1)/Gamma)
		rho = atomics.mixture_density(y, p, T, physics)

		''' Check positivity of computed state. '''
		if np.any(T < 0.) or np.any(p < 0) or np.any(rho < 0):
			raise errors.NotPhysicalError

		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho * y
		UqB[:,:,physics.get_momentum_slice()] = j
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho * y, physics) * T \
			+ (rho * y[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * j * u
		# Update adiabatically compressed/expanded tracer partial densities
		UqB[:,:,5:] *= rho / atomics.rho(arhoVecI)
		#phi_crys = 0.4025 * (1.1 - 0.1 * np.cos(2 * np.pi * 0.25 * t)) # crystal vol / suspension vol
		#chi_water = 0.05055
		#UqB[:,:,5] = rho * (1 - phi_crys) / (1 + chi_water)
		#UqB[:,:,6] = rho * phi_crys

		return UqB


class VelocityInlet1D_gaussianPulse(BCWeakPrescribed):
	'''
	injection of gaussian pulse of different crystal content
	Blah blah blah TODO: copy from MassFluxInlet1D
	'''
	def __init__(self, u:float=1.0, p_chamber:float=100e6,
			T_chamber:float=1e3, trace_arho:float=1e-6, sig:float=4, # 1/36
			yWt:float=0.04, yC:float=0.1, cVFav:float=0.4, 
			cVFamp:float=0.1, tpulse:float=20,
			use_linearized:bool=True, newton_tol:float=1e-7, newton_iter_max=20):
		# Ingest args
		self.u, self.p_chamber, self.T_chamber, self.trace_arho = \
			u, p_chamber, T_chamber, trace_arho
		# Angular frequency of variation
		self.sig, self.cVFav, self.cVFamp, self.tpulse = sig, cVFav, cVFamp, tpulse
		self.yWt, self.yC, self.use_linearized, self.newton_tol, self.newton_iter_max = \
			yWt, yC, use_linearized, newton_tol, newton_iter_max

	def get_linearized_boundary_state(self, physics, UqI, normals, x, t):
		''' Compute a boundary state by replacing Riemann problem with acoustic
		waves, and then approximating the acoustic wave. '''
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

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
		# Approximate desired mass flux
		u = self.u
		
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		gas_volfrac = atomics.gas_volfrac(arhoVecI, TGrid, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid, gas_volfrac, physics)
		rhoGrid = atomics.rho(arhoVecI)
		GammaGrid = atomics.Gamma(arhoVecI, physics)
		cGrid = atomics.sound_speed(GammaGrid, pGrid, rhoGrid, gas_volfrac, physics)
		uGrid = momxI / rhoGrid
		# Compute composition-based properties
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# Outflow
			Gamma = atomics.Gamma(arhoVecI, physics)
			y = atomics.massfrac(arhoVecI)
			S = TGrid / pGrid**((Gamma-1)/Gamma)
		else:
			# Inflow
			arhoVecB = arhoVecI.copy()
			arhoVecB[...,0] = self.trace_arho # Small amount of air to preserve positivity
			arhoVecB[...,1] = self.trace_arho # Small amount of water to preserve positivity
			# Approximate partial density of magma by density
			arhoVecB[...,2] = rho0 * (1 + (p_chamber - p0) / K)
			Gamma = atomics.Gamma(arhoVecB, physics)
			y = atomics.massfrac(arhoVecB)
			S = T_chamber / p_chamber**((Gamma-1)/Gamma)

		# Specific gas constant for R
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Evaluate primitive variables for boundary state
		p = pGrid - (rhoGrid * cGrid) * (uGrid - u)
		T = S * p**((Gamma-1)/Gamma)
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
		phi_crys = self.cVFav + self.cVFamp * np.exp(-((t - self.tpulse)/ self.sig) **2 / 2)
		chi_water = 0.03
		UqB[:,:,5] = rho * chi_water * (1.0 - phi_crys) / (1 + chi_water)
		UqB[:,:,6] = rho * phi_crys
	
		# Fragmented state
		UqB[:,:,7] = 0.0

		return UqB

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		if self.use_linearized:
			# Delegate to linearized
			return self.get_linearized_boundary_state(physics, UqI, normals, x, t)
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		# n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# TODO: improve out flow and sonic handling
			print("Attempting to outflow into an inlet")
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to outgoing acoustic waves) of the interior
		solution. '''
		# Extract data from input and prescribed chamber/reservoir values
		arhoVecI = UqI[:,:,physics.get_mass_slice()]
		momxI = UqI[...,physics.get_momentum_slice()]
		eI = UqI[...,physics.get_state_slice("Energy")]
		j, p_chamber, T_chamber = self.mass_flux, self.p_chamber, self.T_chamber
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		# Compute intermediates
		Gamma = atomics.Gamma(arhoVecI, physics)
		y = atomics.massfrac(arhoVecI)
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Compute chamber entropy as ratio T/p^(..)
		S_r = T_chamber / p_chamber**((Gamma-1)/Gamma)
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid,
			atomics.gas_volfrac(arhoVecI, TGrid, physics), physics)

		def eval_fun_dfun(p):
			''' Evaluate function G and its derivative dG/dp. '''
			# Define reusable groups
			g1 = yRGas * p**(-1/Gamma) * S_r
			g2 = y[...,2] * K / (rho0*(p + K - p0))
			# Integration of 1/impedance
			# Note that the output p, T are not used, since entropy is not taken from grid
			_, uTarget, _ = atomics.velocity_RI_fixed_p_quadrature(p, UqI, physics, normals,
				is_adaptive=True, tol=1e-1, rtol=1e-5)
			# Evaluate integrand
			f = atomics.acousticRI_integrand_scalar(np.array(p), TGrid, pGrid, y, Gamma, physics)
			# Evaluate returns
			G = j * (g1 + g2) + normals * uTarget
			dGdp = -j * (g1 * (1/Gamma) / p + g2 / (p+K-p0)) - f
			return G, dGdp, (p, uTarget)

		# Perform Newton iteration to compute boundary p
		p = pGrid.copy()
		for i in range(self.newton_iter_max):
			G, dGdp, _ = eval_fun_dfun(p)
			p -= G / dGdp
			if np.abs(G) < self.newton_tol:
				break
		# TODO: set logging if max iter is reached
		# Evaluate primitive variables for boundary state
		_, _, (p, u) = eval_fun_dfun(p)
		T = S_r * p**((Gamma-1)/Gamma)
		rho = atomics.mixture_density(y, p, T, physics)

		''' Check positivity of computed state. '''
		if np.any(T < 0.) or np.any(p < 0) or np.any(rho < 0):
			raise errors.NotPhysicalError

		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho * y
		UqB[:,:,physics.get_momentum_slice()] = j
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho * y, physics) * T \
			+ (rho * y[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * j * u
		# Update adiabatically compressed/expanded tracer partial densities
		UqB[:,:,5:] *= rho / atomics.rho(arhoVecI)
		#phi_crys = 0.4025 * (1.1 - 0.1 * np.cos(2 * np.pi * 0.25 * t)) # crystal vol / suspension vol
		#chi_water = 0.05055
		#UqB[:,:,5] = rho * (1 - phi_crys) / (1 + chi_water)
		#UqB[:,:,6] = rho * phi_crys

		return UqB

class MassFluxInlet1D(BCWeakPrescribed):
	'''
	This class corresponds to an outflow boundary condition with static
	incoming mass flux, for a fixed chamber/reservoir pressure and temperature.
	The boundary state is computed using acoustic Riemann invariants in 1D.
	Attributes:
	-----------
	mass_flux:        mass flux at boundary, const
	p_chamber:        pressure of chamber, const (not necessarily bdry value)
	T_chamber:        temperature of chamber, const (not necessarily bdry value)
	newton_tol:       absolute tolerance in residual equation (units of velocity)
	newton_iter_max:  max number of newton iterations to take

	'''
	def __init__(self, mass_flux:float=20e3, p_chamber:float=100e6,
			T_chamber:float=1e3, trace_arho:float=1e-6, yWt:float=0.04, yC:float=0.1,
			use_linearized:bool=True, newton_tol:float=1e-7, newton_iter_max=20):
		# Ingest args
		self.mass_flux, self.p_chamber, self.T_chamber, self.trace_arho = \
			mass_flux, p_chamber, T_chamber, trace_arho
		self.yWt, self.yC, self.use_linearized, self.newton_tol, self.newton_iter_max = \
			yWt, yC, use_linearized, newton_tol, newton_iter_max

	def get_linearized_boundary_state(self, physics, UqI, normals, x, t):
		''' Compute a boundary state by replacing Riemann problem with acoustic
		waves, and then approximating the acoustic wave. '''
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to outgoing acoustic waves) of the interior
		solution. '''
		# Extract data from node values
		arhoVecI = UqI[:,:,physics.get_mass_slice()]
		momxI = UqI[...,physics.get_momentum_slice()]
		eI = UqI[...,physics.get_state_slice("Energy")]
		# Extract specified mass flux and prescribed chamber/reservoir values
		j, p_chamber, T_chamber = self.mass_flux, self.p_chamber, self.T_chamber
		# Extract material properties
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		# Approximate desired mass flux
		u = j / rho0
		
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		gas_volfrac = atomics.gas_volfrac(arhoVecI, TGrid, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid, gas_volfrac, physics)
		rhoGrid = atomics.rho(arhoVecI)
		GammaGrid = atomics.Gamma(arhoVecI, physics)
		cGrid = atomics.sound_speed(GammaGrid, pGrid, rhoGrid, gas_volfrac, physics)
		uGrid = momxI / rhoGrid
		# Compute composition-based properties
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# Outflow
			Gamma = atomics.Gamma(arhoVecI, physics)
			y = atomics.massfrac(arhoVecI)
			S = TGrid / pGrid**((Gamma-1)/Gamma)
		else:
			# Inflow
			arhoVecB = arhoVecI.copy()
			arhoVecB[...,0] = 1e-6 # Small amount of air to prevent oscillations
			arhoVecB[...,1] = 1e-6 # Small amount of water to prevent oscillations
			# Approximate partial density of magma by density
			arhoVecB[...,2] = rho0 * (1 + (p_chamber - p0) / K)
			Gamma = atomics.Gamma(arhoVecB, physics)
			y = atomics.massfrac(arhoVecB)
			S = T_chamber / p_chamber**((Gamma-1)/Gamma)

		# Specific gas constant for R
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Evaluate primitive variables for boundary state
		p = pGrid - (rhoGrid * cGrid) * (uGrid - u)
		T = S * p**((Gamma-1)/Gamma)
		rho = atomics.mixture_density(y, p, T, physics)

		''' Check positivity of computed state. '''
		if np.any(T < 0.) or np.any(p < 0) or np.any(rho < 0):
			raise errors.NotPhysicalError

		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho * y
		UqB[:,:,physics.get_momentum_slice()] = rho0 * u
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho * y, physics) * T \
			+ (rho * y[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * rho0 * u * u
		# Update adiabatically compressed/expanded tracer partial densities
		UqB[:,:,5:6] = self.yWt * rho
		UqB[:,:,6:7] = self.yC * rho
		# UqB[:,:,5:] *= rho / atomics.rho(arhoVecI)

		return UqB

	def get_boundary_state(self, physics, UqI, normals, x, t):
		''' Computes the boundary state that satisfies the pressure BC strongly. '''

		if self.use_linearized:
			# Delegate to linearized
			return self.get_linearized_boundary_state(physics, UqI, normals, x, t)
		UqB = UqI.copy()
		''' Check validity of flow state, check number of boundary points. '''
		# n_hat = normals/np.linalg.norm(normals, axis=2, keepdims=True)
		if np.any(UqI[:, :, physics.get_momentum_slice()] * normals > 0.):
			# TODO: improve out flow and sonic handling
			print("Attempting to outflow into an inlet")
		if UqI.shape[0] * UqI.shape[1] > 1:
			raise NotImplementedError('''Not implemented: for-loop over more than one
				inflow boundary point.''')

		''' Compute boundary-satisfying primitive state that preserves Riemann
		invariants (corresponding to outgoing acoustic waves) of the interior
		solution. '''
		# Extract data from input and prescribed chamber/reservoir values
		arhoVecI = UqI[:,:,physics.get_mass_slice()]
		momxI = UqI[...,physics.get_momentum_slice()]
		eI = UqI[...,physics.get_state_slice("Energy")]
		j, p_chamber, T_chamber = self.mass_flux, self.p_chamber, self.T_chamber
		K, rho0, p0 = \
			physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
		# Compute intermediates
		Gamma = atomics.Gamma(arhoVecI, physics)
		y = atomics.massfrac(arhoVecI)
		yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
		# Compute chamber entropy as ratio T/p^(..)
		S_r = T_chamber / p_chamber**((Gamma-1)/Gamma)
		# Compute grid primitives
		TGrid = atomics.temperature(arhoVecI, momxI, eI, physics)
		pGrid = atomics.pressure(arhoVecI, TGrid,
			atomics.gas_volfrac(arhoVecI, TGrid, physics), physics)

		def eval_fun_dfun(p):
			''' Evaluate function G and its derivative dG/dp. '''
			# Define reusable groups
			g1 = yRGas * p**(-1/Gamma) * S_r
			g2 = y[...,2] * K / (rho0*(p + K - p0))
			# Integration of 1/impedance
			# Note that the output p, T are not used, since entropy is not taken from grid
			_, uTarget, _ = atomics.velocity_RI_fixed_p_quadrature(p, UqI, physics, normals,
				is_adaptive=True, tol=1e-1, rtol=1e-5)
			# Evaluate integrand
			f = atomics.acousticRI_integrand_scalar(np.array(p), TGrid, pGrid, y, Gamma, physics)
			# Evaluate returns
			G = j * (g1 + g2) + normals * uTarget
			dGdp = -j * (g1 * (1/Gamma) / p + g2 / (p+K-p0)) - f
			return G, dGdp, (p, uTarget)

		# Perform Newton iteration to compute boundary p
		p = pGrid.copy()
		for i in range(self.newton_iter_max):
			G, dGdp, _ = eval_fun_dfun(p)
			p -= G / dGdp
			if np.abs(G) < self.newton_tol:
				break
		# TODO: set logging if max iter is reached
		# Evaluate primitive variables for boundary state
		_, _, (p, u) = eval_fun_dfun(p)
		T = S_r * p**((Gamma-1)/Gamma)
		rho = atomics.mixture_density(y, p, T, physics)

		''' Check positivity of computed state. '''
		if np.any(T < 0.) or np.any(p < 0) or np.any(rho < 0):
			raise errors.NotPhysicalError

		''' Map to conservative variables '''
		UqB[:,:,physics.get_mass_slice()] = rho * y
		UqB[:,:,physics.get_momentum_slice()] = j
		UqB[:,:,physics.get_state_slice("Energy")] = \
			atomics.c_v(rho * y, physics) * T \
			+ (rho * y[:,:,2:3]) * physics.Liquid["E_m0"] \
			+ 0.5 * j * u
		# Update adiabatically compressed/expanded tracer partial densities
		UqB[:,:,5:] *= rho / atomics.rho(arhoVecI)
		#phi_crys = 0.4025 * (1.1 - 0.1 * np.cos(2 * np.pi * 0.25 * t)) # crystal vol / suspension vol
		#chi_water = 0.05055
		#UqB[:,:,5] = rho * (1 - phi_crys) / (1 + chi_water)
		#UqB[:,:,6] = rho * phi_crys

		return UqB


class NohInlet(BCWeakRiemann):
	'''
	Inlet conditions for the exact Noh solution (exact for the strong shock limit
	with zero pressure in the unshocked fluid)
	'''

	def __init__(self, eps=1e-3, rho0=1.0, u0=1.0, L=1.0):
		''' Set epsilon for pressure of the unshocked fluid, the density scale rho0,
		the velocity scale u0 > 0 of the converging unshocked fluid (magnitude only)
		and the distance of the boundary from the cylindrical axis.
		'''
		self.eps = eps
		self.rho0 = rho0
		self.u0 = u0
		self.L = L

	def get_boundary_state(self, physics, UqI, normals, x, t):		
		# Compute scalar values of exact unshocked solution for strong shock limit
		rho = self.rho0 * (1.0 + self.u0 * t / self.L)
		# Select larger eps of pressure to correct for nonzero interior pressure
		e = 0.5 * rho * self.u0**2.0 + self.eps / (physics.Gas[0]["gamma"] - 1.0)

		# Package boundary state
		iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()
		UqB = UqI.copy()
		UqB[:, :, iarhoA]  = rho
		UqB[:, :, iarhoWv] = 0.0
		UqB[:, :, iarhoM]  = 0.0
		UqB[:, :, irhou]   = -rho * self.u0
		UqB[:, :, irhov]   = 0.0
		UqB[:, :, ie]      = e
		UqB[:, :, iarhoWt] = 0.0
		UqB[:, :, iarhoC]  = 0.0
		UqB[:, :, iarhoFm]  = 0.0

		return UqB


'''
---------------------
Source term functions
---------------------
These classes inherit from the SourceBase class. See SourceBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the SourceType enum members above.
'''

class CylindricalGeometricSource(SourceBase):
	'''
	Geometric source term that arises when interpreting Cartesian coordinates
	(x,z) as cylindrical coordinates (r,z) under axisymmetric conditions and zero
	momentum in the azimuthal direction (if nonzero, Coriolis and centrifugal
	effects need to be added).
	'''

	def __init__(self, **kwargs):
		super().__init__(kwargs)

	def get_source(self, physics, Uq, x, t):
		''' Returns the geometric source due to divergences in polar (cylindrical)
		coordinates. Interpret the first spatial coordinate as r, and incur the
		geometric source term. '''

		# Interpret first spatial coordinate as radial coordinate r
		r = x[:,:,0:1]
		# Compute 1/r for r > 0
		r_inv = r.copy()
		r_inv[np.nonzero(r_inv)] = 1.0 / r_inv[np.nonzero(r_inv)]
		if np.any(r_inv > 1e10):
			raise Exception("Poorly conditioned mesh: 1/r is large for r not equal to 0.")

		''' Compute radial flux '''
		# TODO: reduce redundant comps, use faster atomics. The fastest way is to
		# re-use the computation of radial flux in the interior, and cache p to
		# remove it from the source term.
		F, (u2, v2, a) = physics.get_conv_flux_interior(Uq)
		# Interpret first coordinate of flux as radial flux (does not copy F)
		F_r = F[:, :, :, 0]
		# Remove pressure contribution (isotropic tensor pI has no contribution to
		# geometric source terms, unlike the momentum dyad)
		iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()
		p = physics.compute_additional_variable("Pressure", Uq, True).squeeze(axis=2)
		F_r[:, :, irhou] -= p

		# Set source term equal to -1/r times the radial flux due to advection
		return -r_inv * F_r

class FrictionVolFracVariableMu(SourceBase):
	'''
	Friction term for a volume fraction fragmentation criterion, equipped with a
	variable viscosity depending on crystal and dissolved water content.
	WITH SMOOTHING

	Attributes:
	-----------
	conduit_radius: conduit radius used in Poiseuille approximation (units m)
	crit_volfrac: critical volume fraction at which friction model transitions (-)
	logistic_scale: scale of logistic function (-)
	'''
	def __init__(self, conduit_radius:float=50.0, crit_volfrac:float=0.8,
							 logistic_scale:float=0.004,**kwargs):
		super().__init__(kwargs)
		self.conduit_radius = conduit_radius
		self.crit_volfrac = crit_volfrac
		self.logistic_scale = logistic_scale

	def compute_indicator(self, phi):
		''' Defines smoothed indicator for turning on friction. Takes value 1
		when friction should be maximized, and value 0 when friction should be off.
		'''
		return 1.0 / (
			1.0 + np.exp((phi - self.crit_volfrac) / self.logistic_scale))
	
	def get_indicator_deriv(self, phi):
		''' Defines derivative of the smoothed indicator.
		'''
		return (1.0/self.logistic_scale) * self.compute_indicator(phi) \
			* (self.compute_indicator(phi) - 1.0)
	
	def compute_viscosity(self, Uq, physics):
		''' calculates the viscosity at each depth point as function of dissolved
		water and crystal content (assumes crystal phase is incompressible)
		'''
		### calculating viscosity of melt without crystals
		### Hess & Dingwell 1996
		temp = atomics.temperature(Uq[..., physics.get_mass_slice()],
			Uq[..., physics.get_momentum_slice()],
			Uq[..., physics.get_state_slice("Energy")],
			physics)
		phi = atomics.gas_volfrac(Uq[..., physics.get_mass_slice()], temp, physics)
		phiM = 1.0 - phi
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()
		arhoWv = Uq[:, :, iarhoWv:iarhoWv+1]
		arhoM  = Uq[:, :, iarhoM:iarhoM+1]
		arhoWt = Uq[:, :, iarhoWt:iarhoWt+1]
		arhoC  = Uq[:, :, iarhoC:iarhoC+1]
		
		arhoWd = arhoWt - arhoWv
		arhoMelt = arhoM - arhoWd - arhoC # partical density of melt ONLY
		mfWd = arhoWd / arhoMelt # mass concentration of dissolved water
		log_mfWd = np.log(mfWd*100)
		
		log10_vis = -3.545 + 0.833 * log_mfWd
		log10_vis += (9601 - 2368 * log_mfWd) / (temp - 195.7 - 32.25 * log_mfWd)
		# log10_vis[(1 - phiM) > self.crit_volfrac] = 0 # turning off friction above fragmentation
		log10_vis = np.where(log10_vis > 300, 300, log10_vis)
		meltVisc = 10**log10_vis
		# limit = np.max(abs(meltVisc))
		# meltVisc[(1 - phiM) > self.crit_volfrac] = limit
		
		### calculating relative viscosity due to crystals
		### Costa 2005b
		alpha = 0.999916
		phi_cr = 0.673
		gamma = 3.98937
		delta = 16.9386
		B = 2.5
		#rhoC = arhoM / phiM #crystal phasic density set to magma phasic density
		# since setting crystal phasic density equal to magma phasic density,
		# crystal vol frac of magma is ratio of partial densities
		# WILL NEED TO CHANGE IF CRYSTAL PHASIC DENSITY DIFFERS
		crysVolFrac_suspension = arhoC / arhoM # crystal vol frac of magma
		
		phi_ratio = crysVolFrac_suspension / phi_cr
		AA = np.sqrt(np.pi) / (2 * alpha)
		erf = sp.erf(AA * phi_ratio * (1 + phi_ratio**gamma))
		
		num = 1 + phi_ratio**delta
		denom = (1 - alpha * erf)**(-B * phi_cr)
		crysVisc = num * denom
		
		#viscosity = 4.386e5 * crysVisc
		viscosity = meltVisc * crysVisc
		#viscosity[(1 - phiM) > self.crit_volfrac] = 0
		
		#fix = np.max(viscosity)
		#viscosity[phi > self.crit_volfrac] = fix
		return viscosity

	def get_source(self, physics, Uq, x, t):
		'''
		Output:
		-----------
		source vector S [ne, nd, ns]
		'''
		if physics.NDIMS != 1:
			raise Exception(f"Conduit friction source not suitable for use in " +
											f"{physics.NDIMS} spatial dimensions.")
		#iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
		#	physics.get_state_indices()

		''' Compute mixture density, u, friction coefficient '''
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2,keepdims=True)
		u = Uq[:, :, physics.get_momentum_slice()] / (rho + general.eps)
		mu = self.compute_viscosity(Uq, physics)
		fric_coeff = 8.0 * mu / self.conduit_radius**2.0
		''' Compute indicator based on proportion of fragmented phase '''
		slarhoFm = physics.get_state_slice("pDensityFm")
		slarhoM = physics.get_state_slice("pDensityM")
		arhoFm = Uq[:,:,slarhoFm]
		arhoM = Uq[:,:,slarhoM]
		I = np.clip(1 - arhoFm / arhoM, 0, 1)
		''' Compute source vector at each element [ne, nq] '''
		S = np.zeros_like(Uq)
		S[:, :, physics.get_momentum_slice()] = -I * fric_coeff * u
		S[:, :, physics.get_state_slice("Energy")] = -I * fric_coeff * u * u
		return S

	def get_phi_gradient():
		''' Compute gradient of total gas volume fraction with respect to state
		vector.

		The gradient of total gas volume fraction (appearing as a negative) is
		needed in gradients of friction terms with volume fraction fragmentation
		criteria, e.g., friction source terms that contain smoothed indicators of
		magma volume fraction. Here magma volume fraction is 1 - phi, and phi is the
		sum of volume fractions of all exsolved gas components.
		'''
		pass

	def get_jacobian(self, physics, Uq, x, t):
		''' Computes the Jacobian of the source vector f_i = s_i I(phi(U)), where I
		is a smoothed indicator dependent on the complete state U. Using the product
		rule, we write
				d_j(f_i) = I * d_j(s_i) + I' * s_i * d_j(phi),
		where d_j is the j-th partial and I' is the ordinary derivative of I.

		Evaluation and inversions of jacobians are likely to be a comp. bottleneck
		(repeated construction for implicit source steps, followed by inversion).
		'''
		raise NotImplementedError(f"Gradient not implemented for variable viscosity.")

		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		phi = physics.compute_additional_variable("phi", Uq, True)

		''' Compute Jacobian of physical expression for friction, times I '''
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2,keepdims=True)
		u = Uq[:, :, physics.get_momentum_slice()] / (rho + general.eps)
		mu = self.compute_viscosity(Uq, physics)
		fric_coeff = 8.0 * mu / self.conduit_radius**2.0		
		friction_jacobian = np.zeros(
			[Uq.shape[0], Uq.shape[1], Uq.shape[-1], Uq.shape[-1]])
		# friction_jacobian[:, :, imom, physics.get_mass_slice()] = u / rho
		# friction_jacobian[:, :, imom, imom] = -1.0 / rho
		# friction_jacobian[:, :, ie, physics.get_mass_slice()] = 2*u**2.0 / rho
		# friction_jacobian[:, :, ie, ie] = -2.0 * u / rho
		# Broadcasted multiplication
		# friction_jacobian *= fric_coeff * self.compute_indicator( \
		# 	physics.compute_additional_variable("phi", Uq, True))
		''' Optimized construction '''
		coeffinvrho = fric_coeff / rho * self.compute_indicator(phi)
		coeffuinvrho = u * coeffinvrho
		friction_jacobian[:, :, imom, physics.get_mass_slice()] = coeffuinvrho
		friction_jacobian[:, :, imom, physics.get_momentum_slice()] = -coeffinvrho
		friction_jacobian[:, :, ie, physics.get_mass_slice()] = 2.0 * u * coeffuinvrho
		friction_jacobian[:, :, ie, physics.get_momentum_slice()] = -2.0 * coeffuinvrho		

		''' Compute Jacobian of indicator, times max amount of friction '''
		friction_vec = self.get_source(physics, Uq, x, t)
		indicator_jacobian = self.get_indicator_deriv(phi) \
			* physics.compute_phi_sgradient(Uq)

		''' Return product derivative '''
		return friction_jacobian + np.einsum('lmi, lmj -> lmij',
			friction_vec, indicator_jacobian)

class FrictionVolFracVariableMu_SHARP(SourceBase):
	'''
	Friction term for a volume fraction fragmentation criterion, equipped with a
	variable viscosity depending on crystal and dissolved water content.

	Attributes:
	-----------
	conduit_radius: conduit radius used in Poiseuille approximation (units m)
	crit_volfrac: critical volume fraction at which friction model transitions (-)
	logistic_scale: scale of logistic function (-)
	'''
	def __init__(self, conduit_radius:float=50.0, crit_volfrac:float=0.8,
							 logistic_scale:float=0.004,**kwargs):
		super().__init__(kwargs)
		self.conduit_radius = conduit_radius
		self.crit_volfrac = crit_volfrac
		self.logistic_scale = logistic_scale

	def compute_indicator(self, phi):
		''' Defines smoothed indicator for turning on friction. Takes value 1
		when friction should be maximized, and value 0 when friction should be off.
		'''
		return 1.0 / (
			1.0 + np.exp((phi - self.crit_volfrac) / self.logistic_scale))
	
	def get_indicator_deriv(self, phi):
		''' Defines derivative of the smoothed indicator.
		'''
		return (1.0/self.logistic_scale) * self.compute_indicator(phi) \
			* (self.compute_indicator(phi) - 1.0)
	
	def compute_viscosity(self, Uq, physics):
		''' calculates the viscosity at each depth point as function of dissolved
		water and crystal content (assumes crystal phase is incompressible)
		'''
		### calculating viscosity of melt without crystals
		### Hess & Dingwell 1996
		temp = physics.compute_additional_variable("Temperature", Uq, True)
		phiM = physics.compute_additional_variable("volFracM", Uq, True)
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = physics.get_state_indices()
		arhoWv = Uq[:, :, iarhoWv:iarhoWv+1]
		arhoM  = Uq[:, :, iarhoM:iarhoM+1]
		arhoWt = Uq[:, :, iarhoWt:iarhoWt+1]
		arhoC  = Uq[:, :, iarhoC:iarhoC+1]
		
		arhoWd = arhoWt - arhoWv
		arhoMelt = arhoM - arhoWd - arhoC # partical density of melt ONLY
		mfWd = arhoWd / arhoMelt # mass concentration of dissolved water
		log_mfWd = np.log(mfWd*100)
		
		log10_vis = -3.545 + 0.833 * log_mfWd
		log10_vis += (9601 - 2368 * log_mfWd) / (temp - 195.7 - 32.25 * log_mfWd)
		log10_vis[(1 - phiM) > self.crit_volfrac] = 0 # turning off friction above fragmentation
		meltVisc = 10**log10_vis
		meltVisc[(1 - phiM) > self.crit_volfrac] = 0
		
		### calculating relative viscosity due to crystals
		### Costa 2005b
		alpha = 0.999916
		phi_cr = 0.673
		gamma = 3.98937
		delta = 16.9386
		B = 2.5
		#rhoC = arhoM / phiM #crystal phasic density set to magma phasic density
		# since setting crystal phasic density equal to magma phasic density,
		# crystal vol frac of magma is ratio of partial densities
		# WILL NEED TO CHANGE IF CRYSTAL PHASIC DENSITY DIFFERS
		crysVolFrac_suspension = arhoC / arhoM # crystal vol frac of magma
		
		phi_ratio = crysVolFrac_suspension / phi_cr
		AA = np.sqrt(np.pi) / (2 * alpha)
		erf = sp.erf(AA * phi_ratio * (1 + phi_ratio**gamma))
		
		num = 1 + phi_ratio**delta
		denom = (1 - alpha * erf)**(-B * phi_cr)
		crysVisc = num * denom
		
		#viscosity = 4.386e5 * crysVisc
		viscosity = meltVisc * crysVisc
		viscosity[(1 - phiM) > self.crit_volfrac] = 0
		
		#fix = np.max(viscosity)
		#viscosity[phi > self.crit_volfrac] = fix
		return viscosity

	def get_source(self, physics, Uq, x, t):
		'''
		Output:
		-----------
		source vector S [ne, nd, ns]
		'''
		if physics.NDIMS != 1:
			raise Exception(f"Conduit friction source not suitable for use in " +
											f"{physics.NDIMS} spatial dimensions.")
		#iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
		#	physics.get_state_indices()

		''' Compute mixture density, u, friction coefficient '''
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2,keepdims=True)
		u = Uq[:, :, physics.get_momentum_slice()] / (rho + general.eps)
		mu = self.compute_viscosity(Uq, physics)
		fric_coeff = 8.0 * mu / self.conduit_radius**2.0
		''' Compute indicator based on proportion of fragmented phase '''
		slarhoFm = physics.get_state_slice("pDensityFm")
		slarhoM = physics.get_state_slice("pDensityM")
		arhoFm = Uq[:,:,slarhoFm]
		arhoM = Uq[:,:,slarhoM]
		I = np.clip(1 - arhoFm / arhoM, 0, 1)
		''' Compute source vector at each element [ne, nq] '''
		S = np.zeros_like(Uq)
		S[:, :, physics.get_momentum_slice()] = -I * fric_coeff * u
		S[:, :, physics.get_state_slice("Energy")] = -I * fric_coeff * u**2.0
		return S

	def get_phi_gradient():
		''' Compute gradient of total gas volume fraction with respect to state
		vector.

		The gradient of total gas volume fraction (appearing as a negative) is
		needed in gradients of friction terms with volume fraction fragmentation
		criteria, e.g., friction source terms that contain smoothed indicators of
		magma volume fraction. Here magma volume fraction is 1 - phi, and phi is the
		sum of volume fractions of all exsolved gas components.
		'''
		pass

	def get_jacobian(self, physics, Uq, x, t):
		''' Computes the Jacobian of the source vector f_i = s_i I(phi(U)), where I
		is a smoothed indicator dependent on the complete state U. Using the product
		rule, we write
				d_j(f_i) = I * d_j(s_i) + I' * s_i * d_j(phi),
		where d_j is the j-th partial and I' is the ordinary derivative of I.

		Evaluation and inversions of jacobians are likely to be a comp. bottleneck
		(repeated construction for implicit source steps, followed by inversion).
		'''
		raise NotImplementedError(f"Gradient not implemented for variable viscosity.")

		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		phi = physics.compute_additional_variable("phi", Uq, True)

		''' Compute Jacobian of physical expression for friction, times I '''
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2,keepdims=True)
		u = Uq[:, :, physics.get_momentum_slice()] / (rho + general.eps)
		mu = self.compute_viscosity(Uq, physics)
		fric_coeff = 8.0 * mu / self.conduit_radius**2.0		
		friction_jacobian = np.zeros(
			[Uq.shape[0], Uq.shape[1], Uq.shape[-1], Uq.shape[-1]])
		# friction_jacobian[:, :, imom, physics.get_mass_slice()] = u / rho
		# friction_jacobian[:, :, imom, imom] = -1.0 / rho
		# friction_jacobian[:, :, ie, physics.get_mass_slice()] = 2*u**2.0 / rho
		# friction_jacobian[:, :, ie, ie] = -2.0 * u / rho
		# Broadcasted multiplication
		# friction_jacobian *= fric_coeff * self.compute_indicator( \
		# 	physics.compute_additional_variable("phi", Uq, True))
		''' Optimized construction '''
		coeffinvrho = fric_coeff / rho * self.compute_indicator(phi)
		coeffuinvrho = u * coeffinvrho
		friction_jacobian[:, :, imom, physics.get_mass_slice()] = coeffuinvrho
		friction_jacobian[:, :, imom, physics.get_momentum_slice()] = -coeffinvrho
		friction_jacobian[:, :, ie, physics.get_mass_slice()] = 2.0 * u * coeffuinvrho
		friction_jacobian[:, :, ie, physics.get_momentum_slice()] = -2.0 * coeffuinvrho		

		''' Compute Jacobian of indicator, times max amount of friction '''
		friction_vec = self.get_source(physics, Uq, x, t)
		indicator_jacobian = self.get_indicator_deriv(phi) \
			* physics.compute_phi_sgradient(Uq)

		''' Return product derivative '''
		return friction_jacobian + np.einsum('lmi, lmj -> lmij',
			friction_vec, indicator_jacobian)

class FrictionVolFracConstMu(SourceBase):
	'''
	Friction term for a volume fraction fragmentation criterion, equipped with a
	constant viscosity.

	Attributes:
	-----------
	mu: viscosity (units Pa s)
	conduit_radius: conduit radius used in Poiseuille approximation (units m)
	crit_volfrac: critical volume fraction at which friction model transitions (-)
	logistic_scale: scale of logistic function (-)
	'''
	def __init__(self, mu:float=1.4e6, conduit_radius:float=50.0,
							 crit_volfrac:float=0.8, logistic_scale:float=0.01,**kwargs):
		super().__init__(kwargs)
		self.mu = mu
		self.conduit_radius = conduit_radius
		self.crit_volfrac = crit_volfrac
		self.logistic_scale = logistic_scale

	def compute_indicator(self, phi):
		''' Defines smoothed indicator for turning on friction. Takes value 1
		when friction should be maximized, and value 0 when friction should be off.
		'''
		return 1.0 / (
			1.0 + np.exp((phi - self.crit_volfrac) / self.logistic_scale))
	
	def get_indicator_deriv(self, phi):
		''' Defines derivative of the smoothed indicator.
		'''
		return (1.0/self.logistic_scale) * self.compute_indicator(phi) \
			* (self.compute_indicator(phi) - 1.0)

	def get_source(self, physics, Uq, x, t):
		'''
		Output:
		-----------
		source vector S [ne, nd, ns]
		'''
		if physics.NDIMS != 1:
			raise Exception(f"Conduit friction source not suitable for use in " +
											f"{physics.NDIMS} spatial dimensions.")
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		''' Compute mixture density, u, friction coefficient '''
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2,keepdims=True)
		u = Uq[:, :, physics.get_momentum_slice()] / (rho + general.eps)
		fric_coeff = 8.0 * self.mu / self.conduit_radius**2.0
		''' Compute indicator based on magma porosity '''
		I = self.compute_indicator( \
			physics.compute_additional_variable("phi", Uq, True))
		''' Compute source vector at each element [ne, nq] '''
		S = np.zeros_like(Uq)
		S[:, :, physics.get_momentum_slice()] = -I * fric_coeff * u
		S[:, :, physics.get_state_slice("Energy")] = -I * fric_coeff * u**2.0
		return S

	def get_phi_gradient():
		''' Compute gradient of total gas volume fraction with respect to state
		vector.

		The gradient of total gas volume fraction (appearing as a negative) is
		needed in gradients of friction terms with volume fraction fragmentation
		criteria, e.g., friction source terms that contain smoothed indicators of
		magma volume fraction. Here magma volume fraction is 1 - phi, and phi is the
		sum of volume fractions of all exsolved gas components.
		'''
		pass

	def get_jacobian(self, physics, Uq, x, t):
		''' Computes the Jacobian of the source vector f_i = s_i I(phi(U)), where I
		is a smoothed indicator dependent on the complete state U. Using the product
		rule, we write
				d_j(f_i) = I * d_j(s_i) + I' * s_i * d_j(phi),
		where d_j is the j-th partial and I' is the ordinary derivative of I.

		Evaluation and inversions of jacobians are likely to be a comp. bottleneck
		(repeated construction for implicit source steps, followed by inversion).
		'''

		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()

		phi = physics.compute_additional_variable("phi", Uq, True)

		''' Compute Jacobian of physical expression for friction, times I '''
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2,keepdims=True)
		u = Uq[:, :, physics.get_momentum_slice()] / (rho + general.eps)
		fric_coeff = 8.0 * self.mu / self.conduit_radius**2.0		
		friction_jacobian = np.zeros(
			[Uq.shape[0], Uq.shape[1], Uq.shape[-1], Uq.shape[-1]])
		# friction_jacobian[:, :, imom, physics.get_mass_slice()] = u / rho
		# friction_jacobian[:, :, imom, imom] = -1.0 / rho
		# friction_jacobian[:, :, ie, physics.get_mass_slice()] = 2*u**2.0 / rho
		# friction_jacobian[:, :, ie, ie] = -2.0 * u / rho
		# Broadcasted multiplication
		# friction_jacobian *= fric_coeff * self.compute_indicator( \
		# 	physics.compute_additional_variable("phi", Uq, True))
		''' Optimized construction '''
		coeffinvrho = fric_coeff / rho * self.compute_indicator(phi)
		coeffuinvrho = u * coeffinvrho
		friction_jacobian[:, :, imom, physics.get_mass_slice()] = coeffuinvrho
		friction_jacobian[:, :, imom, physics.get_momentum_slice()] = -coeffinvrho
		friction_jacobian[:, :, ie, physics.get_mass_slice()] = 2.0 * u * coeffuinvrho
		friction_jacobian[:, :, ie, physics.get_momentum_slice()] = -2.0 * coeffuinvrho		

		''' Compute Jacobian of indicator, times max amount of friction '''
		friction_vec = self.get_source(physics, Uq, x, t)
		indicator_jacobian = self.get_indicator_deriv(phi) \
			* physics.compute_phi_sgradient(Uq)

		''' Return product derivative '''
		return friction_jacobian + np.einsum('lmi, lmj -> lmij',
			friction_vec, indicator_jacobian)


class GravitySource(SourceBase):
	'''
	Gravity source term. Applies gravity for 1D in negative x-direction, and for
	2D in negative y-direction.
	'''
	def __init__(self, gravity=0., **kwargs):
		super().__init__(kwargs)
		self.gravity = gravity

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		# Compute mixture density
		rho = np.sum(Uq[:, :, physics.get_mass_slice()],axis=2)
		if physics.NDIMS == 1:
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			# Orient gravity in axial direction
			S[:, :, imom] = -rho * self.gravity
			S[:, :, ie]   = -Uq[:, :, imom] * self.gravity # rhou * g (gravity work)
		elif physics.NDIMS == 2:
			# Orient gravity in y direction
			iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			S[:, :, irhov] = -rho * self.gravity
			S[:, :, ie] = -Uq[:, :, irhov] * self.gravity
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


class ExsolutionSource(SourceBase):
	'''
	Exsolution source term.
	Equilibrium concentration and dissolution timescales are saved in the physics
	object (those quantities are also used to set up initial conditions).
	Dynamic parameters (tau_d) are attributes of this class.
	'''
	def __init__(self, tau_d:float=1.0, **kwargs):
		super().__init__(kwargs)
		self.tau_d = tau_d

	@staticmethod
	def get_eq_conc(physics, p):
			''' Compute Henry equilibrium concentration '''
			k = physics.Solubility["k"]
			n = physics.Solubility["n"]
			return k * p ** n

	@staticmethod
	def get_eq_conc_deriv(physics, p):
		''' Compute equilibrium concentration derivative'''
		k = physics.Solubility["k"]
		n = physics.Solubility["n"]
		return (k * n) * p ** (n-1)

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		if physics.NDIMS == 1:
			# Extract variables
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			slarhoA = physics.get_state_slice("pDensityA")
			slarhoWv = physics.get_state_slice("pDensityWv")
			slarhoM = physics.get_state_slice("pDensityM")
			slarhoWt = physics.get_state_slice("pDensityWt")
			slarhoC = physics.get_state_slice("pDensityC")
			arhoA = Uq[:, :, slarhoA]
			arhoWv = Uq[:, :, slarhoWv]
			arhoM = Uq[:, :, slarhoM]
			arhoWt = Uq[:, :, slarhoWt]
			arhoC = Uq[:, :, slarhoC]
			p = physics.compute_additional_variable("Pressure", Uq, True)

			# Mixture density
			rho = Uq[:, :, physics.get_mass_slice()].sum(axis=-1, keepdims=True)
			yWt, yA, yC = arhoWt / rho, arhoA / rho, arhoC / rho
			yL = 1.0 - yWt - yA - yC
			# Clipped target vapour mass fraction
			yWvTarget = np.clip(
				yWt - yL * ExsolutionSource.get_eq_conc(physics, p),
				1e-7, 1.0)
			S_scalar = (1.0/self.tau_d) * rho * (
				yWvTarget - arhoWv/rho
			)

			# # Legacy 
			# eq_conc = ExsolutionSource.get_eq_conc(physics, p)
			# S_scalar = (1.0/self.tau_d) * (
			# 	(1.0+eq_conc) * arhoWt 
			# 	- (1.0+eq_conc) * arhoWv
			# 	- eq_conc * (arhoM-arhoC)) # arhoM - arhoC: melt with dissolved water

			# Replace limiting value for absent magma and absent water
			S_scalar[np.where(np.logical_and(
				arhoWt-arhoWv <= general.eps,
				arhoM <= general.eps
			))] = 0.0
			
			# # Legacy
			# # Switch-off of source term for zero exsolved water
			# # Set epsilon below which the source term is quadratic in arhoWv. Must be
			# # large enough to limit stiff sources.
			# quadr_eps = 1e-1
			# # Compute rate factor <= 1 that smoothly goes to zero when arhoWv goes to
			# # zero, but is 1 when arhoWv > quadr_eps
			# rateFactor = np.minimum(arhoWv, quadr_eps) / quadr_eps 
			# S_scalar *= rateFactor**2.0

			S[:, :, slarhoWv] =  S_scalar
			S[:, :, slarhoM]  = -S_scalar
		else:
			raise Exception("Unexpected physics num dimension in GravitySource.")
		return S # [ne, nq, ns]
	
	def get_jacobian(self, physics, Uq, x, t):
		jac = np.zeros([Uq.shape[0], Uq.shape[1], Uq.shape[-1], Uq.shape[-1]])
		if physics.NDIMS == 1:
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			dSdU = self.compute_exsolution_source_sgradient(physics, Uq)
			jac[:, :, iarhoWv, :] = dSdU
			jac[:, :, iarhoM, :] = -dSdU
		else:
			raise Exception("Unexpected physics num dimension in GravitySource.")
		return jac

	def compute_exsolution_source_sgradient(self, physics, Uq):
		'''
		Compute the state-gradient of the exsolution scalar source function.

		Inputs:
		-------
			Uq: solution in each element evaluated at quadrature points [ne, nq, ns]

		Outputs:
		--------
			array: state-gradient of the scalar source function [ne, nq, ns]
		'''
		if physics.NDIMS != 1:
			raise NotImplementedError(f"compute_exsolution_source_sgradient called for" +
																f"NDIMS=={self.NDIMS}, which is not 1.")
		raise NotImplementedError(f"Gradient does not account for cystal content. If" +
			" this is fine, remove the raise in compute_exsolution_source_sgradient. This" +
			" gradient is used typically for implicit source timestepping.")
		
		# Extract variables
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
			physics.get_state_indices()
		slarhoWv = physics.get_state_slice("pDensityWv")
		slarhoM = physics.get_state_slice("pDensityM")
		slarhoWt = physics.get_state_slice("pDensityWt")
		arhoWv = Uq[:, :, slarhoWv]
		arhoM = Uq[:, :, slarhoM]
		arhoWt = Uq[:, :, slarhoWt]
		p = physics.compute_additional_variable("Pressure", Uq, True)

		# Compute source gradient
		dSdU = np.zeros_like(Uq)
		# Change in source term due to pressure-related solubility change
		dSdU = (arhoM - arhoWt + arhoWv) \
			* ExsolutionSource.get_eq_conc_deriv(physics, p) \
			* physics.compute_pressure_sgradient(Uq)
		# Chemical potential change
		dSdU[:, :, slarhoWv] += (1.0+ExsolutionSource.get_eq_conc(physics, p))
		dSdU[:, :, slarhoM] += ExsolutionSource.get_eq_conc(physics, p)
		# Apply tau_d
		dSdU /= -self.tau_d
		# Remove vacuum-related spurious values at quadrature points
		dSdU[np.where(np.logical_and(
			arhoWt-arhoWv <= general.eps,
			arhoM <= general.eps
			))] = 0.0

		return dSdU


class FragmentationTimescaleSourceSmoothed(SourceBase):
	'''
	Fragmentation source term.
	Converts unfragmented magma to fragmented magma over a timescale when
	the fragmentation criterion is met.
	Dynamic parameters (tau_f) are attributes of this class.
	'''
	def __init__(self, tau_f:float=1.0, crit_volfrac:float=0.8,
							 fragsmooth_scale:float=0.010, **kwargs):
		super().__init__(kwargs)
		self.tau_f, self.crit_volfrac, self.fragsmooth_scale = \
			 tau_f, crit_volfrac, fragsmooth_scale

	def smoother(self, x, scale):
		''' Returns one-sided smoothing u(x) of a step, such that
			1. u(x < -scale) = 0
			2. u(x >= 0) = 1.
			3. u smoothly interpolates from 0 to 1 in between.
		'''
		# Shift, scale, and clip to [-1, 0] to prevent exp overflow
		_x = np.clip(x / scale + 1, 0, 1)
		f0 = np.exp(-1/np.where(_x == 0, 1, _x))
		f1 = np.exp(-1/np.where(_x == 1, 1, 1-_x))
		# Return piecewise evaluation
		return np.where(_x >= 1, 1,
						np.where(_x <= 0, 0, 
							f0 / (f0 + f1)))

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		if physics.NDIMS == 1:
			# Extract variables
			slarhoM = physics.get_state_slice("pDensityM")
			slarhoFm = physics.get_state_slice("pDensityFm")

			mom = Uq[:, :, physics.get_momentum_slice()]
			arho = Uq[:, :, physics.get_mass_slice()]
			e = Uq[:, :, physics.get_state_slice("Energy")]
			arhoM = Uq[:, :, slarhoM]
			arhoFm = Uq[:, :, slarhoFm]

			T = atomics.temperature(arho, mom, e, physics)
			phi = atomics.gas_volfrac(arho, T, physics)

			# Compute smoothing argument and smoothed 0-1
			smoothed_0_1 = self.smoother(phi - self.crit_volfrac,
				self.fragsmooth_scale)
			# Compute reaction rate of current state
			reaction_rate = (1.0/self.tau_f) * \
				smoothed_0_1 * (arhoM - arhoFm) # Proportional to non-fragmented mass
			# Apply reaction rate to only fragmented magma (arhom = fragmented + unfragmented)
			S[:, :, slarhoFm]  = reaction_rate
		else:
			raise Exception("Unexpected physics num dimension "
				+ "in FragmentationTimescaleSourceSmoothed.")
		return S # [ne, nq, ns]


class FragmentationTimescaleSource(SourceBase):
	'''
	Fragmentation source term.
	Converts unfragmented magma to fragmented magma over a timescale when
	the fragmentation criterion is met.
	Dynamic parameters (tau_f) are attributes of this class.
	'''
	def __init__(self, tau_f:float=1.0, crit_volfrac:float=0.8, **kwargs):
		super().__init__(kwargs)
		self.tau_f, self.crit_volfrac = tau_f, crit_volfrac

	def get_is_fragmenting(self, physics, arhoVec, T):
			''' Check volume fraction condition. '''
			return (atomics.gas_volfrac(arhoVec, T, physics) > self.crit_volfrac
				).astype(float)

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		if physics.NDIMS == 1:
			# Extract variables
			slarhoM = physics.get_state_slice("pDensityM")
			slarhoFm = physics.get_state_slice("pDensityFm")

			mom = Uq[:, :, physics.get_momentum_slice()]
			arho = Uq[:, :, physics.get_mass_slice()]
			e = Uq[:, :, physics.get_state_slice("Energy")]
			arhoM = Uq[:, :, slarhoM]
			arhoFm = Uq[:, :, slarhoFm]

			T = atomics.temperature(arho, mom, e, physics)
			
			# Compute reaction rate of current state
			reaction_rate = (1.0/self.tau_f) * \
				self.get_is_fragmenting(physics, arho, T) * (
				arhoM - arhoFm) # Proportional to non-fragmented mass
			# Apply reaction rate to only fragmented magma (arhom = fragmented + unfragmented)
			S[:, :, slarhoFm]  = reaction_rate
		else:
			raise Exception("Unexpected physics num dimension in FragmentationTimescaleSource.")
		return S # [ne, nq, ns]


class WaterInflowSource(SourceBase):
	'''
	Water Inflow Source term, equipped with water as an ideal gas
	Inputs:
	-------
		Uq:
	Outputs:
	'''

	def __init__(self, aquifer_depth:float=-500.0, aquifer_length:float=100.0, **kwargs):
		super().__init__(kwargs)
		self.aquifer_depth = aquifer_depth
		self.aquifer_length = aquifer_length

	def get_source(self, physics, Uq, x, t):
		S = np.zeros_like(Uq)
		if physics.NDIMS == 1:
			# Extract variables
			iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm = \
				physics.get_state_indices()
			slarhoWv = physics.get_state_slice("pDensityWv")
			slarhoM = physics.get_state_slice("pDensityM")
			slarhoWt = physics.get_state_slice("pDensityWt")
			sle = physics.get_state_slice("Energy")
			arhoWv = Uq[:, :, slarhoWv]
			arhoM = Uq[:, :, slarhoM]
			arhoWt = Uq[:, :, slarhoWt]
			e = Uq[:, :, sle]

			# formulating j
			darcy_vel = 10 ** -5 * 10 ** 5  # range decided with Eric: 10^-8 - 10^-5
			radius = 50  # in meters & hardcoded
			rho_w = 75 # 10 ** 3  # kg/m^3
			j = darcy_vel * rho_w * (2/radius)  # Starosin has different j value

			# formulating q
			T_w = 290  # K
			# steam_table = XSteam(XSteam.UNIT_SYSTEM_BARE)
			# enthalpy_per_mass = 1e3 * steam_table.h_pt(10, T_w)
			enthalpy_per_mass = physics.Gas[1]["c_p"] * T_w
			q = enthalpy_per_mass * rho_w * darcy_vel * (2 / radius)  #T_w * 4182 * rho_w * darcy_vel

			# variables about x
			xmin = x[len(x) - 1]
			xmax = x[0]
			conduit_size = int(np.abs(xmax + xmin))
			NumElemsX = len(x)
			elem_size = conduit_size/NumElemsX

			index_start = int(-self.aquifer_depth/elem_size)
			for i in range(int(self.aquifer_length/elem_size)):
				if t < 0.1: #  physics.pressure_temp[0, 6] < 0.25:  # while time is less than 1
					# adding water density term for inflowing water
					S[:, :, slarhoWv][index_start + i][0] = j
					# adding energy term for inflowing water
					S[:, :, sle][index_start + i][0] = q
		return S  # [ne, nq, ns]


'''
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class. 
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes 
and methods. Information specific to the corresponding child classes can 
be found below. These classes should correspond to the ConvNumFluxType 
or DiffNumFluxType enum members above.
'''
class LaxFriedrichs1D(ConvNumFluxBase):
	'''
	Local Lax-Friedrichs flux function.
	'''
	def compute_flux(self, physics, UqL, UqR, normals):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Left flux
		FqL, (u2L, aL) = physics.get_conv_flux_projected(UqL, n_hat)

		# Right flux
		FqR, (u2R, aR) = physics.get_conv_flux_projected(UqR, n_hat)

		# Jump
		dUq = UqR - UqL

		# Max wave speeds at each point ||u|| + a
		wL = np.empty(u2L.shape + (1,))
		wR = np.empty(u2R.shape + (1,))
		wL[:, :, 0] = np.sqrt(u2L) + aL
		wR[:, :, 0] = np.sqrt(u2R) + aR

		# Put together
		return 0.5 * n_mag * (FqL + FqR - np.maximum(wL, wR)*dUq)


class LaxFriedrichs2D(ConvNumFluxBase):
	'''
	This class corresponds to the local Lax-Friedrichs flux function for the
	Euler2D class. This replaces the generalized, less efficient version of
	the Lax-Friedrichs flux found in base.
	'''
	def compute_flux(self, physics, UqL, UqR, normals):
		# Normalize the normal vectors
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Left flux
		FqL, (u2L, v2L, aL) = physics.get_conv_flux_projected(UqL,
				n_hat)

		# Right flux
		FqR, (u2R, v2R, aR) = physics.get_conv_flux_projected(UqR,
				n_hat)

		# Jump
		dUq = UqR - UqL

		# Max wave speeds at each point
		wL = np.empty(u2L.shape + (1,))
		wR = np.empty(u2R.shape + (1,))
		wL[:, :, 0] = np.sqrt(u2L + v2L) + aL
		wR[:, :, 0] = np.sqrt(u2R + v2R) + aR

		# Put together
		return 0.5 * n_mag * (FqL + FqR - np.maximum(wL, wR)*dUq)


class Roe1D(ConvNumFluxBase):
	'''
	1D Roe numerical flux. References:
		[1] P. L. Roe, "Approximate Riemann solvers, parameter vectors, and
		difference schemes," Journal of Computational Physics,
		43(2):357–372, 1981.
		[2] J. S. Hesthaven, T. Warburton, "Nodal discontinuous Galerkin
		methods: algorithms, analysis, and applications," Springer Science
		& Business Media, 2007.

	Attributes:
	-----------
	UqL: numpy array
		helper array for left state [nf, nq, ns]
	UqR: numpy array
		helper array for right state [nf, nq, ns]
	vel: numpy array
		helper array for velocity [nf, nq, ndims]
	alphas: numpy array
		helper array: left eigenvectors multipled by dU [nf, nq, ns]
	evals: numpy array
		helper array for eigenvalues [nf, nq, ns]
	R: numpy array
		helper array for right eigenvectors [nf, nq, ns, ns]
	'''
	def __init__(self, Uq=None):
		'''
		This method initializes the attributes.

		Inputs:
		-------
			Uq: values of the state variables (typically at the quadrature
				points) [nf, nq, ns]; used to allocate helper arrays; if None,
				then empty arrays allocated

		Outputs:
		--------
				self: attributes initialized
		'''
		raise NotImplementedError("Roe flux not implemented for multiphase.")
		if Uq is not None:
			n = Uq.shape[0]
			nq = Uq.shape[1]
			ns = Uq.shape[-1]
			ndims = ns - 2
		else:
			n = nq = ns = ndims = 0

		self.UqL = np.zeros_like(Uq)
		self.UqR = np.zeros_like(Uq)
		self.vel = np.zeros([n, nq, ndims])
		self.alphas = np.zeros_like(Uq)
		self.evals = np.zeros_like(Uq)
		self.R = np.zeros([n, nq, ns, ns])

	def rotate_coord_sys(self, smom, Uq, n):
		'''
		This method expresses the momentum vector in the rotated coordinate
		system, which is aligned with the face normal and tangent.

		Inputs:
		-------
			smom: momentum slice
			Uq: values of the state variable (typically at the quadrature
				points) [nf, nq, ns]
			n: normals (typically at the quadrature points) [nf, nq, ndims]

		Outputs:
		--------
				Uq: momentum terms modified
		'''
		Uq[:, :, smom] *= n

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		'''
		This method expresses the momentum vector in the standard coordinate
		system. It "undoes" the rotation above.

		Inputs:
		-------
			smom: momentum slice
			Uq: values of the state variable (typically at the quadrature
				points) [nf, nq, ns]
			n: normals (typically at the quadrature points) [nf, nq, ndims]

		Outputs:
		--------
				Uq: momentum terms modified
		'''
		Uq[:, :, smom] /= n

		return Uq

	def roe_average_state(self, physics, srho, velL, velR, UqL, UqR):
		'''
		This method computes the Roe-averaged variables.

		Inputs:
		-------
			physics: physics object
			srho: density slice
			velL: left velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			velR: right velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			UqL: left state (typically evaluated at the quadrature
				points) [nf, nq, ns]
			UqR: right state (typically evaluated at the quadrature
				points) [nf, nq, ns]

		Outputs:
		--------
				rhoRoe: Roe-averaged density [nf, nq, 1]
				velRoe: Roe-averaged velocity [nf, nq, ndims]
				HRoe: Roe-averaged total enthalpy [nf, nq, 1]
		'''
		rhoL_sqrt = np.sqrt(UqL[:, :, srho])
		rhoR_sqrt = np.sqrt(UqR[:, :, srho])
		HL = physics.compute_variable("TotalEnthalpy", UqL)
		HR = physics.compute_variable("TotalEnthalpy", UqR)

		velRoe = (rhoL_sqrt*velL + rhoR_sqrt*velR)/(rhoL_sqrt+rhoR_sqrt)
		HRoe = (rhoL_sqrt*HL + rhoR_sqrt*HR)/(rhoL_sqrt+rhoR_sqrt)
		rhoRoe = rhoL_sqrt*rhoR_sqrt

		return rhoRoe, velRoe, HRoe

	def get_differences(self, physics, srho, velL, velR, UqL, UqR):
		'''
		This method computes velocity, density, and pressure jumps.

		Inputs:
		-------
			physics: physics object
			srho: density slice
			velL: left velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			velR: right velocity (typically evaluated at the quadrature
				points) [nf, nq, ndims]
			UqL: left state (typically evaluated at the quadrature
				points) [nf, nq, ns]
			UqR: right state (typically evaluated at the quadrature
				points) [nf, nq, ns]

		Outputs:
		--------
				drho: density jump [nf, nq, 1]
				dvel: velocity jump [nf, nq, ndims]
				dp: pressure jump [nf, nq, 1]
		'''
		dvel = velR - velL
		drho = UqR[:, :, srho] - UqL[:, :, srho]
		dp = physics.compute_variable("Pressure", UqR) - \
				physics.compute_variable("Pressure", UqL)

		return drho, dvel, dp

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		'''
		This method computes alpha_i = ith left eigenvector * dU.

		Inputs:
		-------
			c: speed of sound [nf, nq, 1]
			c2: speed of sound squared [nf, nq, 1]
			dp: pressure jump [nf, nq, 1]
			dvel: velocity jump [nf, nq, ndims]
			drho: density jump [nf, nq, 1]
			rhoRoe: Roe-averaged density [nf, nq, 1]

		Outputs:
		--------
				alphas: left eigenvectors multipled by dU [nf, nq, ns]
		'''
		alphas = self.alphas

		alphas[:, :, 0:1] = 0.5/c2*(dp - c*rhoRoe*dvel[:, :, 0:1])
		alphas[:, :, 1:2] = drho - dp/c2
		alphas[:, :, -1:] = 0.5/c2*(dp + c*rhoRoe*dvel[:, :, 0:1])

		return alphas

	def get_eigenvalues(self, velRoe, c):
		'''
		This method computes the eigenvalues.

		Inputs:
		-------
			velRoe: Roe-averaged velocity [nf, nq, ndims]
			c: speed of sound [nf, nq, 1]

		Outputs:
		--------
				evals: eigenvalues [nf, nq, ns]
		'''
		evals = self.evals

		evals[:, :, 0:1] = velRoe[:, :, 0:1] - c
		evals[:, :, 1:2] = velRoe[:, :, 0:1]
		evals[:, :, -1:] = velRoe[:, :, 0:1] + c

		return evals

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		'''
		This method computes the right eigenvectors.

		Inputs:
		-------
			c: speed of sound [nf, nq, 1]
			evals: eigenvalues [nf, nq, ns]
			velRoe: Roe-averaged velocity [nf, nq, ndims]
			HRoe: Roe-averaged total enthalpy [nf, nq, 1]

		Outputs:
		--------
				R: right eigenvectors [nf, nq, ns, ns]
		'''
		R = self.R

		# first row
		R[:, :, 0, 0:2] = 1.; R[:, :, 0, -1] = 1.
		# second row
		R[:, :, 1, 0] = evals[:, :, 0]; R[:, :, 1, 1] = velRoe[:, :, 0]
		R[:, :, 1, -1] = evals[:, :, -1]
		# last row
		R[:, :, -1, 0:1] = HRoe - velRoe[:, :, 0:1]*c;
		R[:, :, -1, 1:2] = 0.5*np.sum(velRoe*velRoe, axis=2, keepdims=True)
		R[:, :, -1, -1:] = HRoe + velRoe[:, :, 0:1]*c

		return R

	def compute_flux(self, physics, UqL_std, UqR_std, normals):
		# Reshape arrays
		n = UqL_std.shape[0]
		nq = UqL_std.shape[1]
		ns = UqL_std.shape[2]
		ndims = ns - 2
		self.UqL_stdL = np.zeros_like(UqL_std)
		self.UqL_stdR = np.zeros_like(UqL_std)
		self.vel = np.zeros([n, nq, ndims])
		self.alphas = np.zeros_like(UqL_std)
		self.evals = np.zeros_like(UqL_std)
		self.R = np.zeros([n, nq, ns, ns])

		# Unpack
		srho = physics.get_state_slice("Density")
		smom = physics.get_momentum_slice()
		gamma = physics.gamma

		# Unit normals
		n_mag = np.linalg.norm(normals, axis=2, keepdims=True)
		n_hat = normals/n_mag

		# Copy values from standard coordinate system before rotating
		UqL = UqL_std.copy()
		UqR = UqR_std.copy()

		# Rotated coordinate system
		UqL = self.rotate_coord_sys(smom, UqL, n_hat)
		UqR = self.rotate_coord_sys(smom, UqR, n_hat)

		# Velocities
		velL = UqL[:, :, smom]/UqL[:, :, srho]
		velR = UqR[:, :, smom]/UqR[:, :, srho]

		# Roe-averaged state
		rhoRoe, velRoe, HRoe = self.roe_average_state(physics, srho, velL,
				velR, UqL, UqR)

		# Speed of sound from Roe-averaged state
		c2 = (gamma - 1.)*(HRoe - 0.5*np.sum(velRoe*velRoe, axis=2,
				keepdims=True))
		if np.any(c2 <= 0.):
			# Non-physical state
			raise errors.NotPhysicalError
		c = np.sqrt(c2)

		# Jumps
		drho, dvel, dp = self.get_differences(physics, srho, velL, velR,
				UqL, UqR)

		# alphas (left eigenvectors multiplied by dU)
		alphas = self.get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		# Eigenvalues
		evals = self.get_eigenvalues(velRoe, c)
		
		# Entropy fix (currently commented as we have yet to decide
		# if this is needed long term)
		# eps = np.zeros_like(evals)
		# eps[:, :, :] = (1e-2 * c)
		# fix = np.argwhere(np.logical_and(evals < eps, evals > -eps))
		# fix_shape = fix[:, 0], fix[:, 1], fix[:, 2]
		# evals[fix_shape] = 0.5 * (eps[fix_shape] + evals[fix_shape]* \
		# 	evals[fix_shape] / eps[fix_shape])
		
		# Right eigenvector matrix
		R = self.get_right_eigenvectors(c, evals, velRoe, HRoe)

		# Form flux Jacobian matrix multiplied by dU
		FRoe = np.einsum('ijkl, ijl -> ijk', R, np.abs(evals)*alphas)

		# Undo rotation
		FRoe = self.undo_rotate_coord_sys(smom, FRoe, n_hat)

		# Left flux
		FL, _ = physics.get_conv_flux_projected(UqL_std, n_hat)

		# Right flux
		FR, _ = physics.get_conv_flux_projected(UqR_std, n_hat)

		return .5*n_mag*(FL + FR - FRoe) # [nf, nq, ns]


class Roe2D(Roe1D):
	'''
	2D Roe numerical flux. This class inherits from the Roe1D class.
	See Roe1D for detailed comments on the attributes and methods.
	In this class, several methods are updated to account for the extra
	dimension.
	'''
	def rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:, :, smom]

		vel[:, :, 0] = np.sum(Uq[:, :, smom]*n, axis=2)
		vel[:, :, 1] = np.sum(Uq[:, :, smom]*n[:, :, ::-1]*np.array([[-1.,
				1.]]), axis=2)

		Uq[:, :, smom] = vel

		return Uq

	def undo_rotate_coord_sys(self, smom, Uq, n):
		vel = self.vel
		vel[:] = Uq[:, :, smom]

		vel[:, :, 0] = np.sum(Uq[:, :, smom]*n*np.array([[1., -1.]]), axis=2)
		vel[:, :, 1] = np.sum(Uq[:, :, smom]*n[:, :, ::-1], axis=2)

		Uq[:, :, smom] = vel

		return Uq

	def get_alphas(self, c, c2, dp, dvel, drho, rhoRoe):
		alphas = self.alphas

		alphas = super().get_alphas(c, c2, dp, dvel, drho, rhoRoe)

		alphas[:, :, 2:3] = rhoRoe*dvel[:, :, -1:]

		return alphas

	def get_eigenvalues(self, velRoe, c):
		evals = self.evals

		evals = super().get_eigenvalues(velRoe, c)

		evals[:, :, 2:3] = velRoe[:, :, 0:1]

		return evals

	def get_right_eigenvectors(self, c, evals, velRoe, HRoe):
		R = self.R

		R = super().get_right_eigenvectors(c, evals, velRoe, HRoe)

		i = 2

		# First row
		R[:, :, 0, i] = 0.
		#  Second row
		R[:, :, 1, i] = 0.
		#  Last (fourth) row
		R[:, :, -1, i] = velRoe[:, :, -1]
		#  Third row
		R[:, :, i, 0] = velRoe[:, :, -1];  R[:, :, i, 1] = velRoe[:, :, -1]
		R[:, :, i, -1] = velRoe[:, :, -1]; R[:, :, i, i] = 1.

		return R
