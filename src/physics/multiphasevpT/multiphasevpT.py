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
#       File : src/physics/multiphasevpT/multiphasevpT.py
#
#       Contains class definitions for 1D and 2D multiphase equations (mixture
#       theory). One classification of this set of equations is the vpT
#       relaxation of the Baer-Nunziato equations with a mixture of two ideal
#       gases and a linearized barotropic material.
#
# ------------------------------------------------------------------------ #
from enum import Enum
import numpy as np

import errors
import general

import physics.base.base as base
import physics.base.functions as base_fcns
from physics.base.functions import BCType as base_BC_type
from physics.base.functions import ConvNumFluxType as base_conv_num_flux_type
from physics.base.functions import FcnType as base_fcn_type

import physics.multiphasevpT.functions as mpvpT_fcns
from physics.multiphasevpT.functions import BCType
from physics.multiphasevpT.functions import ConvNumFluxType
from physics.multiphasevpT.functions import FcnType
from physics.multiphasevpT.functions import SourceType

import numerics.helpers.helpers as helpers
from dataclasses import dataclass

class MultiphasevpT(base.PhysicsBase):
	'''
	This class corresponds to the compressible Euler equations for a
	calorically perfect gas. It inherits attributes and methods from the
	PhysicsBase class. See PhysicsBase for detailed comments of attributes
	and methods. This class should not be instantiated directly. Instead,
	the 1D and 2D variants, which inherit from this class (see below),
	should be instantiated.

	Additional methods and attributes are commented below.

	Attributes:
	-----------
	R: float
		mass-specific gas constant
	gamma: float
		specific heat ratio
	'''
	PHYSICS_TYPE = general.PhysicsType.MultiphasevpT

	def __init__(self, mesh):
		super().__init__(mesh)

	def set_maps(self):
		super().set_maps()

		self.BC_map.update({
			base_BC_type.StateAll : base_fcns.StateAll,
			base_BC_type.Extrapolate : base_fcns.Extrapolate,
			BCType.SlipWall : mpvpT_fcns.SlipWall,
			# BCType.PressureOutlet : mpvpT_fcns.PressureOutlet,
			# BCType.MultiphasevpT2D1D: mpvpT_fcns.MultiphasevpT2D1D,
			# BCType.MultiphasevpT2D2D: mpvpT_fcns.MultiphasevpT2D2D,
		})

	def set_physical_params(self, 
													Gas1={"R": 287., "gamma": 1.4},
													Gas2={"R": 8.314/18.02e-3, "c_p": 2.288e3}, 
													Liquid={"K": 10e9, "rho0": 2.5e3, "p0": 5e6,
																	"E_m0": 0, "c_m": 3e3},
													Solubility={"k": 5e-6, "n": 0.5},
													mu = 3e5,
													tau_d = 0.5):
		'''
		This method sets physical parameters. Given two of the four ideal gas
		properties, this method computes the remaining two. Redundant parameters are
		replaced.

		Inputs:
		-------
			Gas1: dict representing passive ideal gas (air) w/ two gas params
			Gas2: dict representing soluble ideal gas (water vapor) w/ two gas params
			Liquid: dict representing the linearized barotropic liquid
			Solubility: dict with Henry's law parameters
			mu: Constant Poiseuille viscosity
			tau_d: dissolution/exsolution timescale

		Outputs:
		--------
			self: physical parameters set
		'''
		self.Gas = [Gas1, Gas2]
		self.Liquid = Liquid
		self.Solubility = Solubility
		self.mu = mu
		self.tau_d = tau_d
		for i in range(2):
			if self.Gas[i].get("gamma") is not None:
				if self.Gas[i].get("c_v") is not None:
					self.Gas[i]["c_p"] = self.Gas[i]["gamma"] * self.Gas[i]["c_v"]
					self.Gas[i]["R"] = self.Gas[i]["c_p"] - self.Gas[i]["c_v"]
				elif self.Gas[i].get("c_p") is not None:
					self.Gas[i]["c_v"] = self.Gas[i]["c_p"] / self.Gas[i]["gamma"]
					self.Gas[i]["R"] = self.Gas[i]["c_p"] - self.Gas[i]["c_v"]
				else:
					self.Gas[i]["c_v"] = self.Gas[i]["R"] / (self.Gas[i]["gamma"] - 1.)
					self.Gas[i]["c_p"] = self.Gas[i]["gamma"] * self.Gas[i]["c_v"]
			elif self.Gas[i].get("R") is not None:
				if self.Gas[i].get("c_v") is not None:
					self.Gas[i]["c_p"] = self.Gas[i]["c_v"] + self.Gas[i]["R"]
					self.Gas[i]["gamma"] = self.Gas[i]["c_p"] / self.Gas[i]["c_v"]
				elif self.Gas[i].get("c_p") is not None:
					self.Gas[i]["c_v"] = self.Gas[i]["c_p"] - self.Gas[i]["R"]
					self.Gas[i]["gamma"] = self.Gas[i]["c_p"] / self.Gas[i]["c_v"]
			else:
				self.Gas[i]["gamma"] = self.Gas[i]["c_p"] / self.Gas[i]["c_v"]
				self.Gas[i]["R"] = self.Gas[i]["c_p"] - self.Gas[i]["c_v"]

	class AdditionalVariables(Enum):
		Pressure = "p"
		Temperature = "T"
		Entropy = "s"
		InternalEnergy = "\\rho e"
		TotalEnthalpy = "H"
		SoundSpeed = "c"
		MaxWaveSpeed = "\\lambda"
		Velocity = "|u|"
		XVelocity = "u"
		YVelocity = "v"
		Psi1 = "Psi1"
		beta = "beta"
		pi1 = "pi1"
		pi2 = "pi2"
		pi3 = "pi3"
		Enthalpy = "Enthalpy"
		Gamma = "Gamma"

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		arhoA = Uq[:, :, self.get_state_slice("pDensityA")]
		arhoWv = Uq[:, :, self.get_state_slice("pDensityWv")]
		arhoM = Uq[:, :, self.get_state_slice("pDensityM")]
		mom = Uq[:, :, self.get_momentum_slice()]
		e = Uq[:, :, self.get_state_slice("Energy")]

		''' Flag non-physical state '''
		if flag_non_physical:
			if np.any(arhoA < 0.) or np.any(arhoWv < 0.) or np.any(arhoM < 0.):
				raise errors.NotPhysicalError

		''' Nested functions for common quantities '''
		# Captures states arhoA, ..., mom, e
		def get_temperature():
			kinetic = 0.5*np.sum(mom*mom, axis=2, keepdims=True) \
				/(arhoA + arhoWv + arhoM)
			c_mix = arhoA * self.Gas[0]["c_v"] \
						+ arhoWv * self.Gas[1]["c_v"] \
						+ arhoM * self.Liquid["c_m"]
			return (e - arhoM * self.Liquid["E_m0"] - kinetic)/c_mix
		def get_porosity(T=None):
			if T is None:
				T = get_temperature()
			sym1 = self.Liquid["K"] - self.Liquid["p0"]
			sym2 = (arhoA * self.Gas[0]["R"] + arhoWv * self.Gas[1]["R"]) * T
			return 0.5 / sym1 * (
					sym1 - sym2 - self.Liquid["K"] / self.Liquid["rho0"] * arhoM
					+ np.sqrt(
							np.power((sym1 - sym2 - self.Liquid["K"] / self.Liquid["rho0"] * arhoM
							),2) + 4 * sym1 * sym2)
			)
		def get_pressure(T=None,phi=None):
			if T is None:
				T = get_temperature()
			if phi is None:
				phi = get_porosity(T)
			p = arhoA * self.Gas[0]["R"] * T \
				+ arhoWv * self.Gas[1]["R"] * T \
				+ (1.-phi)*(self.Liquid["p0"] - self.Liquid["K"]) \
					+ (self.Liquid["K"] / self.Liquid["rho0"]) * arhoM
			if flag_non_physical:
				if np.any(p < 0.):
					raise errors.NotPhysicalError
			return p
		def get_Psi1(T=None,phi=None,p=None):
			if T is None:
				T = get_temperature()
			if phi is None:
				phi = get_porosity(T)
			if p is None:
				p = get_pressure(T, phi)
			return (p + self.Liquid["K"] - self.Liquid["p0"]) / ( 
				p + phi * (self.Liquid["K"] - self.Liquid["p0"]))
		def get_Gamma():
			c_mix = arhoA * self.Gas[0]["c_v"] \
						+ arhoWv * self.Gas[1]["c_v"] \
						+ arhoM * self.Liquid["c_m"]
			return 1. + (arhoA * self.Gas[0]["R"] + arhoWv * self.Gas[1]["R"]) / c_mix

		''' Compute '''
		vname = self.AdditionalVariables[var_name].name

		if vname is self.AdditionalVariables["Pressure"].name:
			varq = get_pressure()
		elif vname is self.AdditionalVariables["Temperature"].name:
			varq = get_temperature()
		# elif vname is self.AdditionalVariables["Entropy"].name:
			# varq = R*(gamma/(gamma-1.)*np.log(getT()) - np.log(getP()))
			# Alternate way
			# varq = np.log(get_pressure()/rho**gamma)
		elif vname is self.AdditionalVariables["InternalEnergy"].name:
			varq = e - 0.5*np.sum(mom*mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["Enthalpy"].name:
			varq = e \
				- 0.5*np.sum(mom*mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM) \
				+ get_pressure() / (arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["TotalEnthalpy"].name:
			varq = (e + get_pressure())/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["SoundSpeed"].name:
			varq = np.sqrt(get_Gamma()*get_pressure()/(arhoA+arhoWv+arhoM)*get_Psi1())
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# |u| + c
			varq = np.linalg.norm(mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM) \
				 + np.sqrt(get_Gamma()*get_pressure()/(arhoA+arhoWv+arhoM)*get_Psi1())
		elif vname is self.AdditionalVariables["Velocity"].name:
			varq = np.linalg.norm(mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["XVelocity"].name:
			varq = mom[:, :, [0]]/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["YVelocity"].name:
			varq = mom[:, :, [1]]/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["beta"].name:
			varq = 0.5*(get_Gamma() - 1.) * get_Psi1()
		elif vname is self.AdditionalVariables["Gamma"].name:
			varq = get_Gamma()
		elif vname is self.AdditionalVariables["Psi1"].name:
			varq = get_Psi1()
		elif vname is self.AdditionalVariables["pi1"].name:
			T = get_temperature()
			varq = get_Psi1(T) * (self.Gas[0]["R"] * T
													 - (get_Gamma() - 1) * (self.Gas[0]["c_v"] * T))
		elif vname is self.AdditionalVariables["pi2"].name:
			T = get_temperature()
			varq = get_Psi1(T) * (self.Gas[1]["R"] * T
													 - (get_Gamma() - 1) * (self.Gas[1]["c_v"] * T))
		elif vname is self.AdditionalVariables["pi3"].name:
			T = get_temperature()
			p = get_pressure()
			rhoM = (p - self.Liquid["p0"] + self.Liquid["K"])/self.Liquid["K"]*self.Liquid["rho0"]
			varq = get_Psi1(T=T,p=p) * (p / rhoM
													 - (get_Gamma() - 1) * (
														 self.Liquid["c_m"] * T + self.Liquid["E_m0"]))
		else:
			raise NotImplementedError

		return varq

	def compute_pressure_gradient(self, Uq, grad_Uq):
		'''
		Compute the gradient of pressure with respect to physical space. This is
		needed for pressure-based shock sensors.

		Inputs:
		-------
			Uq: solution in each element evaluated at quadrature points
			[ne, nq, ns]
			grad_Uq: gradient of solution in each element evaluted at quadrature
				points [ne, nq, ns, ndims]

		Outputs:
		--------
			array: gradient of pressure with respected to physical space
				[ne, nq, ndims]
		'''
		slarhoA = self.get_state_slice("pDensityA")
		slarhoWv = self.get_state_slice("pDensityWv")
		slarhoM = self.get_state_slice("pDensityM")
		slmom = self.get_momentum_slice()
		sle = self.get_state_slice("Energy")
		
		arhoA = Uq[:, :, slarhoA]
		arhoWv = Uq[:, :, slarhoWv]
		arhoM = Uq[:, :, slarhoM]
		mom = Uq[:, :, slmom]
		e = Uq[:, :, sle]

		# Retrieve auxiliary quantities
		beta = self.compute_additional_variable("beta", Uq, True)
		bu2 = beta*np.sum(mom**2, axis=2, keepdims=True)/np.power(
			arhoA + arhoWv + arhoM,2.)
		dpdU = np.empty_like(Uq)
		dpdU[:, :, slarhoA] = bu2 + self.compute_additional_variable("pi1", Uq, True)
		dpdU[:, :, slarhoWv] = bu2 + self.compute_additional_variable("pi2", Uq, True)
		dpdU[:, :, slarhoM] = bu2 + self.compute_additional_variable("pi3", Uq, True)
		dpdU[:, :, slmom] = -2.*beta*mom/(arhoA + arhoWv + arhoM)
		dpdU[:, :, sle] = 2.*beta
		

		# Compute dp/dU
		# dpdU = np.empty_like(Uq)
		# dpdU[:, :, srho] = (.5 * (gamma - 1) * np.sum(mom**2, axis = 2,
		# 	keepdims=True) / rho) # rho^2?
		# dpdU[:, :, smom] = (1 - gamma) * mom / rho
		# dpdU[:, :, srhoE] = gamma - 1

		# Multiply with dU/dx
		return np.einsum('ijk, ijkl -> ijl', dpdU, grad_Uq)


class MultiphasevpT1D(MultiphasevpT):
	'''
	This class corresponds to 1D vpT-equations for a two-gas, one liquid mixture.
	'''
	NUM_STATE_VARS = 5
	NDIMS = 1

	def set_maps(self):
		super().set_maps()

		d = {
			FcnType.RiemannProblem: mpvpT_fcns.RiemannProblem,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			# SourceType.Exsolution: mpvpT_fcns.Exsolution,
			# SourceType.FrictionVolFrac: mpvpT_fcns.FrictionVolFrac,
			# SourceType.GravitySource: mpvpT_fcns.PorousSource,
		})

		self.conv_num_flux_map.update({
			base_conv_num_flux_type.LaxFriedrichs: mpvpT_fcns.LaxFriedrichs1D,
		})

	class StateVariables(Enum):
		pDensityA = "\\alpha_a \\rho_a"
		pDensityWv = "\\alpha_\{wv\} \\rho_wv"
		pDensityM = "\\alpha_m \\rho_m"
		XMomentum = "\\rho u"
		Energy = "\\rho E"

	def get_state_indices(self):
		iarhoA = self.get_state_index("pDensityA")
		iarhoWv = self.get_state_index("pDensityWv")
		iarhoM = self.get_state_index("pDensityM")
		imom = self.get_state_index("XMomentum")
		ie = self.get_state_index("Energy")
		return iarhoA, iarhoWv, iarhoM, imom, ie

	def get_state_slices(self):
		slarhoA = self.get_state_slice("pDensityA")
		slarhoWv = self.get_state_slice("pDensityWv")
		slarhoM = self.get_state_slice("pDensityM")
		slmom = self.get_state_slice("XMomentum")
		sle = self.get_state_slice("Energy")
		return slarhoA, slarhoWv, slarhoM, slmom, sle

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		smom = slice(irhou, irhou+1)
		return smom

	def get_conv_flux_interior(self, Uq):
		# Get indices of state variables
		iarhoA, iarhoWv, iarhoM, imom, ie = self.get_state_indices()
		# Get slices for state variables
		slarhoA = self.get_state_slice("pDensityA")
		slarhoWv = self.get_state_slice("pDensityWv")
		slarhoM = self.get_state_slice("pDensityM")
		slmom = self.get_momentum_slice()
		sle = self.get_state_slice("Energy")
		# Extract data of size # [n, nq]
		arhoA  = Uq[:, :, iarhoA]
		arhoWv = Uq[:, :, iarhoWv]
		arhoM  = Uq[:, :, iarhoM]
		mom    = Uq[:, :, imom]
		e      = Uq[:, :, ie]

		# Compute mixture (total) density
		rho = arhoA + arhoWv + arhoM
		p = self.compute_additional_variable("Pressure", Uq, True)
		p = np.squeeze(p, axis=2)
		# Compute velocity vector
		u = mom / rho
		# Compute squared norm of velocity
		u2 = u**2.
		# u2 = np.sum(mom**2, axis=2, keepdims=True) / np.power(rho, 2.)
		
		# Construct physical flux
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:, :, iarhoA,  0] = arhoA * u
		F[:, :, iarhoWv, 0] = arhoWv * u
		F[:, :, iarhoM,  0] = arhoM * u
		F[:, :, imom,    0] = rho * u2 + p # Flux of momentum
		F[:, :, ie,      0] = (e + p) * u        # Flux of energy
		# Compute sound speed
		a = self.compute_additional_variable("SoundSpeed", Uq, True)
		a = np.squeeze(a, axis=2)

		return F, (u2, a)

	def get_conv_eigenvectors(self, U_bar):
		'''
		This function defines the convective eigenvectors for the
		1D euler equations. This is used with the WENO limiter to
		transform the system of equations from physical space to
		characteristic space.

		Inputs:
		-------
			U_bar: Average state [ne, 1, ns]

		Outputs:
		--------
			right_eigen: Right eigenvector matrix [ne, 1, ns, ns]
			left_eigen: Left eigenvector matrix [ne, 1, ns, ns]
		'''

		# Unpack
		ne = U_bar.shape[0]
		ns = self.NUM_STATE_VARS

		iarhoA, iarhoWv, iarhoM, irhou, ie = self.get_state_indices()

		arhoA  = U_bar[:, :, iarhoA]
		arhoWv = U_bar[:, :, iarhoWv]
		arhoM  = U_bar[:, :, iarhoM]
		rhou   = U_bar[:, :, irhou]
		e      = U_bar[:, :, ie]

		rho = arhoA + arhoWv + arhoM
		u = rhou / rho
		u2 = u**2
		p = np.squeeze(self.compute_additional_variable("Pressure", U_bar, True), axis=2)
		H = (e + p)/rho
		a = np.squeeze(self.compute_additional_variable("SoundSpeed", U_bar, True), axis=2)
		y1 = arhoA/rho
		y2 = arhoWv/rho
		y3 = arhoM/rho
		pi1 = np.squeeze(self.compute_additional_variable("pi1", U_bar, True), axis=2)
		pi2 = np.squeeze(self.compute_additional_variable("pi2", U_bar, True), axis=2)
		pi3 = np.squeeze(self.compute_additional_variable("pi3", U_bar, True), axis=2)
		beta = np.squeeze(self.compute_additional_variable("beta", U_bar, True), axis=2)

		# Allocate the right and left eigenvectors
		right_eigen = np.zeros([ne, 1, ns, ns])
		left_eigen = np.zeros([ne, 1, ns, ns])

		# # Calculate the right and left eigenvectors
		right_eigen[:, :, iarhoA,  iarhoA]  = pi2 - pi3
		right_eigen[:, :, iarhoWv, iarhoA]  = pi3 - pi1
		right_eigen[:, :, iarhoM,  iarhoA]  = pi1 - pi2
		right_eigen[:, :, irhou,   iarhoA]  = 0.
		right_eigen[:, :, ie,      iarhoA]  = 0.
		right_eigen[:, :, iarhoA,  iarhoWv] = -pi2
		right_eigen[:, :, iarhoWv, iarhoWv] = pi1
		right_eigen[:, :, iarhoM,  iarhoWv] = 0.
		right_eigen[:, :, irhou,   iarhoWv] = u*(pi1 - pi2)
		right_eigen[:, :, ie,      iarhoWv] = 0.5*u2*(pi1 - pi2)
		right_eigen[:, :, iarhoA,  iarhoM]  = -2*beta
		right_eigen[:, :, iarhoWv, iarhoM]  = 2*beta
		right_eigen[:, :, iarhoM,  iarhoM]  = 0.
		right_eigen[:, :, irhou,   iarhoM]  = 0.
		right_eigen[:, :, ie,      iarhoM]  = pi1 - pi2
		right_eigen[:, :, iarhoA,  irhou]   = y1
		right_eigen[:, :, iarhoWv, irhou]   = y2
		right_eigen[:, :, iarhoM,  irhou]   = y3
		right_eigen[:, :, irhou,   irhou]   = u - a
		right_eigen[:, :, ie,      irhou]   = H - a*u
		right_eigen[:, :, iarhoA,  ie]      = y1
		right_eigen[:, :, iarhoWv, ie]      = y2
		right_eigen[:, :, iarhoM,  ie]      = y3
		right_eigen[:, :, irhou,   ie]      = u + a
		right_eigen[:, :, ie,      ie]      = H + a*u

		left_eigen = np.linalg.inv(right_eigen)
		# Can uncomment line below to test l dot r = kronecker delta
		# test = np.einsum('elij,eljk->elik', left_eigen, right_eigen)

		return right_eigen, left_eigen # [ne, 1, ns, ns]


class MultiphasevpT2D(MultiphasevpT):
	'''
	This class corresponds to 2D Euler equations for a calorically
	perfect gas. It inherits attributes and methods from the Euler class.
	See Euler for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 6
	NDIMS = 2

	def __init__(self, mesh):
		super().__init__(mesh)

	def set_maps(self):
		super().set_maps()

		d = {
			euler_fcn_type.IsentropicVortex : euler_fcns.IsentropicVortex,
			euler_fcn_type.TaylorGreenVortex : euler_fcns.TaylorGreenVortex,
			euler_fcn_type.GravityRiemann : euler_fcns.GravityRiemann,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			euler_source_type.StiffFriction : euler_fcns.StiffFriction,
			euler_source_type.TaylorGreenSource :
					euler_fcns.TaylorGreenSource,
			euler_source_type.GravitySource : euler_fcns.GravitySource,
		})

		self.conv_num_flux_map.update({
			base_conv_num_flux_type.LaxFriedrichs :
				euler_fcns.LaxFriedrichs2D,
			euler_conv_num_flux_type.Roe : euler_fcns.Roe2D,
		})

	class StateVariables(Enum):
		pDensityA = "\\alpha_a \\rho_a"
		pDensityWv = "\\alpha_\{wv\} \\rho_wv"
		pDensityM = "\\alpha_m \\rho_m"
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Energy = "\\rho E"

	def get_state_indices(self):
		irho = self.get_state_index("Density")
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		irhoE = self.get_state_index("Energy")

		return irho, irhou, irhov, irhoE

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		smom = slice(irhou, irhov + 1)

		return smom

	def get_conv_flux_interior(self, Uq):
		# Get indices/slices of state variables
		irho, irhou, irhov, irhoE = self.get_state_indices()
		smom = self.get_momentum_slice()

		rho  = Uq[:, :, irho]  # [n, nq]
		rhou = Uq[:, :, irhou] # [n, nq]
		rhov = Uq[:, :, irhov] # [n, nq]
		rhoE = Uq[:, :, irhoE] # [n, nq]
		mom  = Uq[:, :, smom]  # [n, nq, ndims]

		# Get velocity in each dimension
		u = rhou / rho
		v = rhov / rho
		# Get squared velocities
		u2 = u**2
		v2 = v**2

		# Calculate pressure using the Ideal Gas Law
		p = (self.gamma - 1.)*(rhoE - 0.5 * rho * (u2 + v2)) # [n, nq]
		# Get off-diagonal momentum
		rhouv = rho * u * v
		# Get total enthalpy
		H = rhoE + p

		# Assemble flux matrix
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		F[:,:,irho,  :] = mom          # Flux of mass in all directions
		F[:,:,irhou, 0] = rho * u2 + p # x-flux of x-momentum
		F[:,:,irhov, 0] = rhouv        # x-flux of y-momentum
		F[:,:,irhou, 1] = rhouv        # y-flux of x-momentum
		F[:,:,irhov, 1] = rho * v2 + p # y-flux of y-momentum
		F[:,:,irhoE, 0] = H * u        # x-flux of energy
		F[:,:,irhoE, 1] = H * v        # y-flux of energy

		return F, (u2, v2, rho, p)
