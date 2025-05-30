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
from logging import warning
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
import physics.multiphasevpT.atomics as atomics

import numerics.helpers.helpers as helpers
import numerics.differentiation.eval_divu as eval_divu
from dataclasses import dataclass

class MultiphasevpT(base.PhysicsBase):
	'''
	This class corresponds to a set of multiphase equations for:
		0. mass per total volume, air
		1. mass per total volume, water, exsolved
		2. mass per total volume, magma
		3. momentum per total volume
		4. total energy per total volume, including kinetic
		----
		5. mass per total volume, water, dissolved
		6. mass of crystals per total volume (crystallinity)
		7. mass of fragmented magma per total volume (11/28/2022 Update)
		----
		8. slip of melt and plug.
	where states 5-7 are essentially passive tracers for the
	conservation equations (pressure and sound speed are not dependent on these
	states), though these tracer states can enter in the source term.

	Abstract class. Inherited by MultiphasevpT1D and MultiphasevpT2D.
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
			BCType.PressureOutlet : mpvpT_fcns.PressureOutlet,
			BCType.Inlet : mpvpT_fcns.Inlet,
			BCType.MultiphasevpT2D2D: mpvpT_fcns.MultiphasevpT2D2D,
			BCType.MultiphasevpT1D1D: mpvpT_fcns.MultiphasevpT1D1D,
			BCType.MultiphasevpT2D1D: mpvpT_fcns.MultiphasevpT2D1D,
			BCType.NonReflective1D : mpvpT_fcns.NonReflective1D,
			BCType.PressureOutlet1D : mpvpT_fcns.PressureOutlet1D,
			BCType.MassFluxInlet1D: mpvpT_fcns.MassFluxInlet1D,
			BCType.PressureStableLinearizedInlet1D: mpvpT_fcns.PressureStableLinearizedInlet1D,
			BCType.PressureStableLinearizedInlet1D_genericFunc: mpvpT_fcns.PressureStableLinearizedInlet1D_genericFunc,
			BCType.VelocityInlet1D: mpvpT_fcns.VelocityInlet1D,
			BCType.VelocityInlet1D_neutralSinusoid: mpvpT_fcns.VelocityInlet1D_neutralSinusoid,
			BCType.VelocityInlet1D_gaussianPulse: mpvpT_fcns.VelocityInlet1D_gaussianPulse,
			BCType.LinearizedImpedance2D: mpvpT_fcns.LinearizedImpedance2D,
			BCType.NohInlet: mpvpT_fcns.NohInlet,
			BCType.NohInletMixture: mpvpT_fcns.NohInletMixture,
			BCType.ChokedInlet2D: mpvpT_fcns.ChokedInlet2D,
			BCType.LinearizedIsothermalOutflow2D: mpvpT_fcns.LinearizedIsothermalOutflow2D,
			BCType.OscillatingSphere: mpvpT_fcns.OscillatingSphere,
		})

	def set_physical_params(self, 
													Gas1={"R": 287., "gamma": 1.4},
													Gas2={"R": 8.314/18.02e-3, "c_p": 2.288e3}, 
													Liquid={"K": 10e9, "rho0": 2.7e3, "p0": 5e6,
																	"E_m0": 0, "c_m": 1e3},
													Solubility={"k": 5e-6, "n": 0.5},
													Viscosity={"mu0": 3e5},
													tau_d = 0.5):
		'''
		This method sets physical parameters. Gas1 and Gas2 are ideal gas parameters
		(R, gamma, c_v, c_p). Given two of the four ideal gas properties, this
		method computes the other two.
		properties, this method computes the remaining two. Redundant parameters are
		replaced.

		Inputs:
		-------
			Gas1: dict representing passive ideal gas (air) w/ two gas params
			Gas2: dict representing soluble ideal gas (water vapor) w/ two gas params
			Liquid: dict representing the linearized barotropic liquid
			Solubility: dict with Henry's law parameters
			Viscosity: dict with viscosity parameters (mu0 for constant-Poiseuille)
			tau_d: dissolution/exsolution timescale

		Outputs:
		--------
			self: physical parameters set
		'''
		self.Gas = [Gas1, Gas2]
		self.Liquid = Liquid
		self.Solubility = Solubility
		self.Viscosity = Viscosity
		self.tau_d = tau_d

		''' Compute remaining ideal gas parameters for Gas1, Gas2'''
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

	def update_div(self, solver, U):
		''' This is a hack.
		Updates divergence in self:PhysicsType to pass in elements of solver. Does
		not use solver.state_coeffs (uses U instead, to accommodate multistage
		time steppers). '''
		
		# Evaluate strain rate at quadrature points
		self.strain_rate = eval_divu.eval_strainrate(solver,
			solver.state_coeffs, 
			solver.elem_helpers.x_elems,
			solver.time)

	class AdditionalVariables(Enum):
		Pressure = "p"
		Temperature = "T"
		Entropy = "s"
		InternalEnergy = "\\rho e"
		TotalEnthalpy = "H" # per mass
		SoundSpeed = "c"
		MaxWaveSpeed = "\\lambda"
		Velocity = "|u|"
		XVelocity = "u"
		YVelocity = "v"
		# Intermediate variables to simplify eigensystem
		Psi1 = "Psi1" # Thermodynamic parameter (liquid-related)
		beta = "beta" # Thermodynamic parameter
		pi1 = "pi1"   # Thermodynamic parameter (Psi1 * Gas1-related quantity)
		pi2 = "pi2"   # Thermodynamic parameter (Psi1 * Gas2-related quantity)
		pi3 = "pi3"   # Thermodynamic parameter (Psi1 * Liquid-related quantity)
		# Thermodynamic quantities
		Enthalpy = "Enthalpy" # Enthalpy *excluding* kinetic energy contribution
		Gamma = "Gamma"       # Pseudogas Gamma (mixture heat capacity ratio)
		phi = "phi"                   # Sum of gas volume fractions
		volFracA = "\\alpha_a"				# Volume fraction, air
		volFracWv = "\\alpha_\{wv\}"	# Volume fraction, exsolved water vapour
		volFracM = "\\alpha_m"				# Volume fraction, magma
		# ---- Kate addition ----
		Drag = "f" # friction term in momentum balance; proportional to wall shear stress

	def compute_additional_variable(self, var_name, Uq, flag_non_physical):
		''' Extract state variables '''
		arhoA = Uq[:, :, self.get_state_slice("pDensityA")]
		arhoWv = Uq[:, :, self.get_state_slice("pDensityWv")]
		arhoM = Uq[:, :, self.get_state_slice("pDensityM")]
		mom = Uq[:, :, self.get_momentum_slice()]
		e = Uq[:, :, self.get_state_slice("Energy")]
		arhoWt = Uq[:, :, self.get_state_slice("pDensityWt")]
		try:
			arhoC = Uq[:, :, self.get_state_slice("pDensityC")]
		except KeyError:
			arhoC = np.zeros_like(arhoA)
		try:
			arhoFm = Uq[:, :, self.get_state_slice("pDensityFm")]
		except KeyError:
			arhoFm = np.zeros_like(arhoA)
		try:
			rhoSlip = Uq[:, :, self.get_state_slice("rhoSlip")]
		except KeyError:
			rhoSlip = np.zeros_like(arhoA)

		''' Flag non-physical state
		The EOS-constrained phases (A, Wv, M) are checked for positivity. Total
		water content, which is used in dissolution/exsolution, is also checked.
		'''
		if flag_non_physical:
			if np.any(arhoA < 0.) or np.any(arhoWv < 0.) or np.any(arhoM < 0.) \
				 or np.any(arhoWt < 0.):
				raise errors.NotPhysicalError

		''' Nested functions for common quantities 
		Common routines that may be called for computing several outputs.
		States arhoA, mom, e, etc. are captured by nested functions at this point.
		'''
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
		# ---- Kate addition ----
		def get_Drag():
			f = self.source_terms[1].get_source(self, Uq, None, None)
			return f[:,:,self.get_momentum_slice()]

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
			p = get_pressure(T=T)
			rhoM = (p - self.Liquid["p0"] + self.Liquid["K"])/self.Liquid["K"]*self.Liquid["rho0"]
			varq = get_Psi1(T=T,p=p) * (p / rhoM
													 - (get_Gamma() - 1) * (
														 self.Liquid["c_m"] * T + self.Liquid["E_m0"]))
		elif vname is self.AdditionalVariables["phi"].name:
			varq = get_porosity()
		elif vname is self.AdditionalVariables["volFracA"].name:
			phi = get_porosity()
			# Compute gas partial pressures except where total gas mass is zero
			ppA = arhoA[np.where(phi > 0)] * self.Gas[0]["R"]
			ppWv = arhoWv[np.where(phi > 0)] * self.Gas[1]["R"]
			varq = phi
			varq[np.where(phi > 0)] = phi[np.where(phi > 0)] * (ppA / (ppA + ppWv))
		elif vname is self.AdditionalVariables["volFracWv"].name:
			phi = get_porosity()
			# Compute gas partial pressures except where total gas mass is zero
			ppA = arhoA[np.where(phi > 0)] * self.Gas[0]["R"]
			ppWv = arhoWv[np.where(phi > 0)] * self.Gas[1]["R"]
			varq = phi
			varq[np.where(phi > 0)] = phi[np.where(phi > 0)] * (ppWv / (ppA + ppWv))
		elif vname is self.AdditionalVariables["volFracM"].name:
			varq = 1.0 - get_porosity()
		elif vname is self.AdditionalVariables["Drag"].name:
			varq = get_Drag()
		elif vname is self.AdditionalVariables["SpecificEntropy"].name:
			rho = (arhoA+arhoWv+arhoM)
			y1 = arhoA / rho
			y2 = arhoWv / rho
			y3 = arhoM / rho
			T = get_temperature()
			p = get_pressure(T=T)
			gammaA = self.Gas[0]["gamma"]
			gammaWv = self.Gas[1]["gamma"]
			varq = y1 * self.Gas[0]["c_p"] * np.log(T / p**((gammaA-1.0)/gammaA)) + \
					y2 * self.Gas[1]["c_p"] * np.log(T / p**((gammaWv-1.0)/gammaWv)) + \
					y3 * self.Liquid["c_m"] * np.log(T)
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
		# Multiply dp/dU with dU/dx
		return np.einsum('ijk, ijkl -> ijl',
			self.compute_pressure_sgradient(Uq), grad_Uq)

	def compute_pressure_sgradient(self, Uq):
		'''
		Compute the state-gradient of pressure.
		The spatial gradient pressure is a different function.

		Inputs:
		-------
			Uq: solution in each element evaluated at quadrature points [ne, nq, ns]

		Outputs:
		--------
			array: gradient of pressure with respected to state [ne, nq, ns]
		'''
		# Extract quantities that determine pressure
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
		dpdU = np.zeros_like(Uq)
		dpdU[:, :, slarhoA] = bu2 + self.compute_additional_variable("pi1", Uq, True)
		dpdU[:, :, slarhoWv] = bu2 + self.compute_additional_variable("pi2", Uq, True)
		dpdU[:, :, slarhoM] = bu2 + self.compute_additional_variable("pi3", Uq, True)
		dpdU[:, :, slmom] = -2.*beta*mom/(arhoA + arhoWv + arhoM)
		dpdU[:, :, sle] = 2.*beta

		return dpdU
	
	def compute_phi_sgradient(self, Uq):
		'''
		Compute the gradient of porosity phi with respect to the state variables.
		This is needed for source gradients (for backward stepping of sources
		involving volume-fraction-based fragmentation criteria).

		This is a gradient with respect to the state vector. A commented line
		computes the gradient with respect to space.

		Inputs:
		-------
			Uq: solution in each element evaluated at quadrature points [ne, nq, ns]

		Outputs:
		--------
			array: gradient of phi with respected to state [ne, nq, ns]
		'''
		if self.NDIMS != 1:
			raise NotImplementedError(f"compute_phi_gradient called for" +
																f"NDIMS=={self.NDIMS}, which is not 1.")
		# Extract quantities
		slarhoA = self.get_state_slice("pDensityA")
		slarhoWv = self.get_state_slice("pDensityWv")
		slarhoM = self.get_state_slice("pDensityM")
		slmom = self.get_momentum_slice()
		sle = self.get_state_slice("Energy")
		# Retrieve auxiliary quantities
		Gamma = self.compute_additional_variable("Gamma", Uq, True)
		phi = self.compute_additional_variable("phi", Uq, True)
		T = self.compute_additional_variable("Temperature", Uq, True)
		unorm = self.compute_additional_variable("Velocity", Uq, True)
		unorm2 = unorm * unorm
		p = self.compute_additional_variable("Pressure", Uq, True)
		u = self.compute_additional_variable("XVelocity", Uq, True)
		denom = p + phi*(self.Liquid["K"] - self.Liquid["p0"])
		coeff = (1/denom) * (1.0-phi)

		dphidU = np.zeros_like(Uq)
		dphidU[:, :, slarhoA] =  coeff * (self.Gas[0]["R"]*T + \
			(Gamma-1)*(0.5*unorm2 - self.Gas[0]["c_v"]*T))
		dphidU[:, :, slarhoWv] = coeff * (self.Gas[1]["R"]*T + \
			(Gamma-1)*(0.5*unorm2 - self.Gas[1]["c_v"]*T))
		dphidU[:, :, slarhoM] = coeff * ((Gamma-1)*(0.5*unorm2 - \
			 self.Liquid["c_m"]*T - self.Liquid["E_m0"])) - \
				 phi*(self.Liquid["K"]/self.Liquid["rho0"])
		dphidU[:, :, slmom] = -coeff * (Gamma-1) * u
		dphidU[:, :, sle] = coeff * (Gamma-1)

		# N. B. Multiply with dU/dx for spatial gradient (function of grad_Uq)
		#   dphidx = np.einsum('ijk, ijkl -> ijl', dphidU, grad_Uq)
		return dphidU

class MultiphasevpT1D(MultiphasevpT):
	'''
	This class corresponds to 1D vpT-equations for a two-gas, one liquid mixture.
	'''
	NUM_STATE_VARS = 9 # Total number of state variables
	NDIMS = 1

	def set_maps(self):
		super().set_maps()

		d = {
			FcnType.RiemannProblem: mpvpT_fcns.RiemannProblem,
			FcnType.UniformExsolutionTest: mpvpT_fcns.UniformExsolutionTest,
			FcnType.LinearPressureGrad: mpvpT_fcns.LinearPressureGrad,
			FcnType.UniformTest: mpvpT_fcns.UniformTest,
			FcnType.RightTravelingGaussian: mpvpT_fcns.RightTravelingGaussian,
			FcnType.SteadyState: mpvpT_fcns.SteadyState,
			FcnType.StaticPlug: mpvpT_fcns.StaticPlug,
			FcnType.NohProblem: mpvpT_fcns.NohProblem,
			FcnType.NohProblemMixture: mpvpT_fcns.NohProblemMixture,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			SourceType.FrictionVolFracVariableMu: mpvpT_fcns.FrictionVolFracVariableMu,
			SourceType.FrictionVolFracConstMu: mpvpT_fcns.FrictionVolFracConstMu,
			SourceType.GravitySource: mpvpT_fcns.GravitySource,
			SourceType.ExsolutionSource: mpvpT_fcns.ExsolutionSource,
			SourceType.FragmentationTimescaleSource: mpvpT_fcns.FragmentationTimescaleSource,
			SourceType.FragmentationTimescaleSourceSmoothed: mpvpT_fcns.FragmentationTimescaleSourceSmoothed,
			SourceType.FragmentationStrainRateSource: mpvpT_fcns.FragmentationStrainRateSource,
			SourceType.SlipSource: mpvpT_fcns.SlipSource,
			SourceType.FrictionVolSlip: mpvpT_fcns.FrictionVolSlip,
			SourceType.WaterInflowSource: mpvpT_fcns.WaterInflowSource,
			SourceType.CylindricalGeometricSource: mpvpT_fcns.CylindricalGeometricSource,
		})

		self.conv_num_flux_map.update({
			base_conv_num_flux_type.LaxFriedrichs: mpvpT_fcns.LaxFriedrichs1D,
		})

	class StateVariables(Enum):
		pDensityA = "\\alpha_a \\rho_a"
		pDensityWv = "\\alpha_\{wv\} \\rho_\{wv\}"
		pDensityM = "\\alpha_m \\rho_m"
		XMomentum = "\\rho u"
		Energy = "e"
		pDensityWt = "\\alpha_\{wt\} \\rho_\{wt\}"
		pDensityC = "\\alpha_c \\rho_c"
		pDensityFm = "\\alpha_\{fm\} \\rho_\{fm\}"
		rhoSlip = "\\rho \\s_x" # Identifier for new phase (slip here for example), assigned a name string for built-in plotting tools

	def get_state_indices(self):
		iarhoA = self.get_state_index("pDensityA")
		iarhoWv = self.get_state_index("pDensityWv")
		iarhoM = self.get_state_index("pDensityM")
		imom = self.get_state_index("XMomentum")
		ie = self.get_state_index("Energy")
		iarhoWt = self.get_state_index("pDensityWt")
		iarhoC = self.get_state_index("pDensityC")
		iarhoFm = self.get_state_index("pDensityFm")
		irhoslip = self.get_state_index("rhoSlip")
		return iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm, irhoslip

	def get_state_slices(self):
		slarhoA = self.get_state_slice("pDensityA")
		slarhoWv = self.get_state_slice("pDensityWv")
		slarhoM = self.get_state_slice("pDensityM")
		slmom = self.get_state_slice("XMomentum")
		sle = self.get_state_slice("Energy")
		slarhoWt = self.get_state_slice("pDensityWt")
		slarhoC = self.get_state_slice("pDensityC")
		slarhoFm = self.get_state_slice("pDensityFm")
		slrhoslip = self.get_state_slice("rhoSlip")
		return slarhoA, slarhoWv, slarhoM, slmom, sle, slarhoWt, slarhoC, slarhoFm, slrhoslip

	def get_mass_slice(self):
		# Get mass component indices of phases
		mass_indices = [self.get_state_index("pDensityA"),
										self.get_state_index("pDensityWv"),
										self.get_state_index("pDensityM")]
		# TODO: suggestion: implement for non-contiguous states
		return slice(np.min(mass_indices), np.min(mass_indices)+len(mass_indices))

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		smom = slice(irhou, irhou+1)
		return smom

	def get_conv_flux_interior(self, Uq):
		# Uncomment to check for correct inverse eigenvector matrix
		# 1. L * R == I
		# print(np.abs(np.einsum("ijkl,ijlm->ijkm",
		# 	self.get_eigenvectors_L(np.tile(Uq,(1,1,1))),
		# 	self.get_eigenvectors_R(np.tile(Uq,(1,1,1)))) - 
		# 	np.eye(7)).max())
		# 2. L == R^{-1}
		# print(np.abs(
		# 	self.get_eigenvectors_L(np.tile(Uq,(1,1,1))) -
		# 	np.linalg.inv(self.get_eigenvectors_R(np.tile(Uq,(1,1,1))))).max())
		# 3. L * R == I subject to conditioning of R
		# print(np.abs(np.einsum("ijkl,ijlm->ijkm",
		# 	self.get_eigenvectors_L(np.tile(Uq,(1,1,1))),
		# 	self.get_eigenvectors_R(np.tile(Uq,(1,1,1)))) - 
		# 	np.eye(7)).max() / np.linalg.cond(self.get_eigenvectors_R(np.tile(Uq,(1,1,1)))))

		# Extract data of size # [n, nq]
		mom    = Uq[:, :, self.get_momentum_slice()]
		e      = Uq[:, :, self.get_state_slice("Energy")]

		# Compute mixture (total) density
		rho = Uq[:, :, self.get_mass_slice()].sum(axis=2, keepdims=True)
		T = atomics.temperature(Uq[:, :, self.get_mass_slice()],
			Uq[:, :, self.get_momentum_slice()],
			Uq[:, :, self.get_state_slice("Energy")],
			self)
		gas_volfrac = atomics.gas_volfrac(Uq[:, :, self.get_mass_slice()], T, self)
		p = atomics.pressure(Uq[:, :, self.get_mass_slice()], T, gas_volfrac, self)
		u = mom / rho
		
		# Construct physical flux
		iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC, iarhoFm, irhoslip = \
			self.get_state_indices()
		F = np.repeat(Uq[...,np.newaxis], self.NDIMS, axis=-1)
		F[:, :, iarhoA,  0:] *= u
		F[:, :, iarhoWv, 0:] *= u
		F[:, :, iarhoM,  0:] *= u
		F[:, :, imom,    0:] = rho * u * u + p
		F[:, :, ie,      0:] = (e + p) * u
		F[:, :, iarhoWt, 0:] *= u
		F[:, :, iarhoC,  0:] *= u
		F[:, :, iarhoFm, 0:] *= u
		F[:, :, irhoslip, 0:] *= u   # Set flux equal to velocity times this state variable
		# Compute sound speed
		a = atomics.sound_speed(
			atomics.Gamma(Uq[:, :, self.get_mass_slice()], self),
			p, rho, gas_volfrac, self)

		return F, ((u*u).squeeze(axis=2), a.squeeze(axis=2))

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

		# Skip legacy code and reroute to analytic eigenvector matrix
		return self.get_eigenvectors_R(U_bar), self.get_eigenvectors_L(U_bar)

	def get_essential_eigenvectors_R(self, U):
		ns_ess = self.NDIMS + 4

		iarhoA, iarhoWv, iarhoM, irhou, ie, _, _, _ = self.get_state_indices()

		''' Compute required variables in squeezed shape [ne, nb,].
		Requires are: y1, y2, y3, pi1, pi2, pi3, beta, u, a, H
		TODO: reuse intermediates to lower comp. cost
		''' 
		rho = U[:,:,self.get_mass_slice()].sum(axis=2)
		p = np.squeeze(self.compute_additional_variable("Pressure", U, True), axis=2)

		y1 = U[:, :, iarhoA]/rho
		y2 = U[:, :, iarhoWv]/rho
		y3 = 1.0 - y1 - y2
		pi1 = np.squeeze(self.compute_additional_variable("pi1", U, True), axis=2)
		pi2 = np.squeeze(self.compute_additional_variable("pi2", U, True), axis=2)
		pi3 = np.squeeze(self.compute_additional_variable("pi3", U, True), axis=2)
		beta = np.squeeze(self.compute_additional_variable("beta", U, True), axis=2)
		u = U[:, :, irhou] / rho
		a = np.squeeze(self.compute_additional_variable("SoundSpeed", U, True), axis=2)
		H = (U[:, :, ie] + p)/rho
		
		right_eigen = np.zeros([U.shape[0], U.shape[1], ns_ess, ns_ess])
		right_eigen[:, :, iarhoA,  iarhoA]  = pi2 - pi3
		right_eigen[:, :, iarhoWv, iarhoA]  = pi3 - pi1
		right_eigen[:, :, iarhoM,  iarhoA]  = pi1 - pi2
		right_eigen[:, :, irhou,   iarhoA]  = 0.
		right_eigen[:, :, ie,      iarhoA]  = 0.
		right_eigen[:, :, iarhoA,  iarhoWv] = -pi2
		right_eigen[:, :, iarhoWv, iarhoWv] = pi1
		right_eigen[:, :, iarhoM,  iarhoWv] = 0.
		right_eigen[:, :, irhou,   iarhoWv] = u*(pi1 - pi2)
		right_eigen[:, :, ie,      iarhoWv] = 0.5*u*u*(pi1 - pi2)
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

		return right_eigen

	def get_essential_eigenvectors_L(self, U):
		ns_ess = self.NDIMS + 4

		iarhoA, iarhoWv, iarhoM, irhou, ie, _, _, _ = self.get_state_indices()

		''' Compute required variables in squeezed shape [ne, nb,].
		Requires are: y1, y2, y3, pi1, pi2, pi3, beta, u, a, H
		TODO: reuse intermediates to lower comp. cost
		''' 
		rho = U[:,:,self.get_mass_slice()].sum(axis=2)
		p = np.squeeze(self.compute_additional_variable("Pressure", U, True), axis=2)

		y1 = U[:, :, iarhoA]/rho
		y2 = U[:, :, iarhoWv]/rho
		y3 = 1.0 - y1 - y2
		pi1 = np.squeeze(self.compute_additional_variable("pi1", U, True), axis=2)
		pi2 = np.squeeze(self.compute_additional_variable("pi2", U, True), axis=2)
		pi3 = np.squeeze(self.compute_additional_variable("pi3", U, True), axis=2)
		beta = np.squeeze(self.compute_additional_variable("beta", U, True), axis=2)
		u = U[:, :, irhou] / rho
		a = np.squeeze(self.compute_additional_variable("SoundSpeed", U, True), axis=2)
		H = (U[:, :, ie] + p)/rho
		
		# Compute intermediate variables
		Pi = y1*pi1 + y2*pi2 + y3*pi3
		f1 = beta*u*u + pi1
		f2 = beta*u*u + pi2
		f3 = beta*u*u + pi3
		u2Pi = 0.5*u*u*Pi
		uHDiff = 0.5*u*u - H

		left_eigen = np.zeros([U.shape[0], U.shape[1], ns_ess, ns_ess])
		left_eigen[:, :, iarhoA,  iarhoA]  = -y3*f1
		left_eigen[:, :, iarhoA, iarhoWv]  = -y3*f2
		left_eigen[:, :, iarhoA,  iarhoM]  = a*a-y3*f3
		left_eigen[:, :, iarhoA,   irhou]  = 2*beta*u*y3
		left_eigen[:, :, iarhoA,      ie]  = -2*beta*y3
		left_eigen[:, :, iarhoWv,  iarhoA] = a*a - f1
		left_eigen[:, :, iarhoWv, iarhoWv] = a*a - f2
		left_eigen[:, :, iarhoWv,  iarhoM] = a*a - f3
		left_eigen[:, :, iarhoWv,   irhou] = 2*beta*u
		left_eigen[:, :, iarhoWv,      ie] = -2*beta
		left_eigen[:, :, iarhoM,  iarhoA]  = u2Pi + pi1*uHDiff
		left_eigen[:, :, iarhoM, iarhoWv]  = u2Pi + pi2*uHDiff
		left_eigen[:, :, iarhoM,  iarhoM]  = u2Pi + pi3*uHDiff
		left_eigen[:, :, iarhoM,   irhou]  = -u*Pi
		left_eigen[:, :, iarhoM,      ie]  = Pi
		left_eigen[:, :, irhou,  iarhoA]   = 0.5*(f1+a*u)
		left_eigen[:, :, irhou, iarhoWv]   = 0.5*(f2+a*u)
		left_eigen[:, :, irhou,  iarhoM]   = 0.5*(f3+a*u)
		left_eigen[:, :, irhou,   irhou]   = -0.5*a - beta*u
		left_eigen[:, :, irhou,      ie]   = beta
		left_eigen[:, :, ie,  iarhoA]      = 0.5*(f1-a*u)
		left_eigen[:, :, ie, iarhoWv]      = 0.5*(f2-a*u)
		left_eigen[:, :, ie,  iarhoM]      = 0.5*(f3-a*u)
		left_eigen[:, :, ie,   irhou]      = 0.5*a - beta*u
		left_eigen[:, :, ie,      ie]      = beta

		# Manually apply factored-mass row-wise
		m1 = np.expand_dims(1.0/(a*a*(pi1-pi2)),axis=-1)
		m2 = np.expand_dims(1.0/(a*a),axis=-1)
		left_eigen[:, :, iarhoA,  :] *= m1
		left_eigen[:, :, iarhoWv, :] *= m1
		left_eigen[:, :, iarhoM,  :] *= m1
		left_eigen[:, :, irhou,   :] *= m2
		left_eigen[:, :, ie,      :] *= m2

		return left_eigen

	def get_eigenvectors_R(self, U):
		ns_ess = self.NDIMS + 4
		ns = self.NUM_STATE_VARS
		# Number of columns for eigenvalue u in essential system
		ns_u = ns_ess - 2

		rho = U[:,:,self.get_mass_slice()].sum(axis=2)
		u = U[:,:,self.get_state_index("XMomentum")] / rho
		a = self.compute_additional_variable("SoundSpeed", U, True).squeeze(axis=2)

		right_eigen = np.zeros([U.shape[0], U.shape[1], ns, ns])
		# Block fill-in
		right_eigen[:, :, 0:ns_ess, 0:ns_ess] = self.get_essential_eigenvectors_R(U)
		right_eigen[:, :, ns_ess:, ns_ess:] = np.eye(ns-ns_ess, ns-ns_ess)
		# Fill-in u-c and u+c columns
		right_eigen[:, :, ns_ess, ns_u] = U[:,:,self.get_state_index("pDensityWt")]/rho * (a-u)/a
		right_eigen[:, :, ns_ess, ns_u+1] = U[:,:,self.get_state_index("pDensityWt")]/rho * (a+u)/a
		right_eigen[:, :, ns_ess+1, ns_u] = U[:,:,self.get_state_index("pDensityC")]/rho * (a-u)/a
		right_eigen[:, :, ns_ess+1, ns_u+1] = U[:,:,self.get_state_index("pDensityC")]/rho * (a+u)/a
		right_eigen[:, :, ns_ess+2, ns_u] = U[:,:,self.get_state_index("pDensityFm")]/rho * (a-u)/a
		right_eigen[:, :, ns_ess+2, ns_u+1] = U[:,:,self.get_state_index("pDensityFm")]/rho * (a+u)/a

		return right_eigen

	def get_eigenvectors_L(self, U):
		ns_ess = self.NDIMS + 4
		ns = self.NUM_STATE_VARS
		# Number of columns for eigenvalue u in essential system
		ns_u = ns_ess - 2

		rho = U[:,:,self.get_mass_slice()].sum(axis=2)
		a2 = self.compute_additional_variable("SoundSpeed", U, True).squeeze(axis=2)**2
		beta = self.compute_additional_variable("beta", U, True).squeeze(axis=2)
		u = U[:,:,self.get_state_index("XMomentum")] / rho
		pi1 = self.compute_additional_variable("pi1", U, True).squeeze(axis=2)
		pi2 = self.compute_additional_variable("pi2", U, True).squeeze(axis=2)
		pi3 = self.compute_additional_variable("pi3", U, True).squeeze(axis=2)
		f0 = (beta-1)*u*u
		f1 = f0 + pi1
		f2 = f0 + pi2
		f3 = f0 + pi3

		# Compute eigenvector matrix of essential system (ne, nb, ns_ess, ns_ess)
		L_ess = self.get_essential_eigenvectors_L(U)

		# Block fill-in complete matrix
		left_eigen = np.zeros([U.shape[0], U.shape[1], ns, ns])
		left_eigen[:, :, 0:ns_ess, 0:ns_ess] = L_ess
		left_eigen[:, :, ns_ess:, ns_ess:] = np.eye(ns-ns_ess, ns-ns_ess)
		
		# Fill-in u-c and u+c columns
		# TODO: replace with operator-form matvec multiplication
		left_eigen[:, :, ns_ess, 0:ns_ess] = \
			np.expand_dims(-U[:,:,self.get_state_index("pDensityWt")]/(rho*a2), axis=2) \
			* np.concatenate(
				tuple(np.expand_dims(v,axis=2) for v in 
					[f1, f2, f3, (1-2*beta)*u, 2*beta,]),
				axis=2
			)
		left_eigen[:, :, ns_ess+1, 0:ns_ess] = \
			np.expand_dims(-U[:,:,self.get_state_index("pDensityC")]/(rho*a2), axis=2) \
			* np.concatenate(
				tuple(np.expand_dims(v,axis=2) for v in 
					[f1, f2, f3, (1-2*beta)*u, 2*beta,]),
				axis=2
			)
		left_eigen[:, :, ns_ess+2, 0:ns_ess] = \
			np.expand_dims(-U[:,:,self.get_state_index("pDensityFm")]/(rho*a2), axis=2) \
			* np.concatenate(
				tuple(np.expand_dims(v,axis=2) for v in 
					[f1, f2, f3, (1-2*beta)*u, 2*beta,]),
				axis=2
			)

		return left_eigen
	
	def get_eigenvalues(self, U):
		pass

class MultiphasevpT2D(MultiphasevpT):
	'''
	This class corresponds to 2D Euler equations for a calorically
	perfect gas. It inherits attributes and methods from the Euler class.
	See Euler for detailed comments of attributes and methods.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = MultiphasevpT1D.NUM_STATE_VARS + 1
	NDIMS = 2

	def __init__(self, mesh):
		super().__init__(mesh)

	def set_maps(self):
		super().set_maps()

		d = {
			FcnType.IsothermalAtmosphere: mpvpT_fcns.IsothermalAtmosphere,
			FcnType.LinearAtmosphere: mpvpT_fcns.LinearAtmosphere,
			FcnType.NohProblem: mpvpT_fcns.NohProblem,
			FcnType.NohProblemMixture: mpvpT_fcns.NohProblemMixture,
			FcnType.UniformAir: mpvpT_fcns.UniformAir,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			SourceType.GravitySource: mpvpT_fcns.GravitySource,
			SourceType.CylindricalGeometricSource: mpvpT_fcns.CylindricalGeometricSource,
		})

		self.conv_num_flux_map.update({
			base_conv_num_flux_type.LaxFriedrichs:
				mpvpT_fcns.LaxFriedrichs2D,
		})

	class StateVariables(Enum):
		pDensityA = "\\alpha_a \\rho_a"
		pDensityWv = "\\alpha_\{wv\} \\rho_\{wv\}"
		pDensityM = "\\alpha_m \\rho_m"
		XMomentum = "\\rho u"
		YMomentum = "\\rho v"
		Energy = "e"
		pDensityWt = "\\alpha_\{wt\} \\rho_\{wt\}"
		pDensityC = "\\alpha_c \\rho_c"
		pDensityFm = "\\alpha_\{fm\} \\rho_\{fm\}"
		rhoSlip = "\\rho \\slip_x" # Identifier for new phase (slip here for example), assigned a name string for built-in plotting tools

	def get_state_indices(self):
		iarhoA = self.get_state_index("pDensityA")
		iarhoWv = self.get_state_index("pDensityWv")
		iarhoM = self.get_state_index("pDensityM")
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		ie = self.get_state_index("Energy")
		iarhoWt = self.get_state_index("pDensityWt")
		iarhoC = self.get_state_index("pDensityC")
		iarhoFm = self.get_state_index("pDensityFm")
		irhoslip = self.get_state_index("rhoSlip")
		return iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm, irhoslip

	def get_mass_slice(self):
		# Get mass component indices
		mass_indices = [self.get_state_index("pDensityA"),
										self.get_state_index("pDensityWv"),
										self.get_state_index("pDensityM")]
		return slice(np.min(mass_indices), np.min(mass_indices)+len(mass_indices))

	def get_momentum_slice(self):
		irhou = self.get_state_index("XMomentum")
		irhov = self.get_state_index("YMomentum")
		smom = slice(irhou, irhov + 1)
		return smom

	def get_conv_flux_interior(self, Uq):
		# Get indices/slices of state variables
		iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm, irhoslip = \
			self.get_state_indices()
		smom = self.get_momentum_slice()
		# Extract data of size [n, nq]
		arhoA  = Uq[:, :, iarhoA]
		arhoWv = Uq[:, :, iarhoWv]
		arhoM  = Uq[:, :, iarhoM]
		rhou   = Uq[:, :, irhou]
		rhov   = Uq[:, :, irhov]
		e      = Uq[:, :, ie]
		arhoWt = Uq[:, :, iarhoWt]
		arhoC  = Uq[:, :, iarhoC]
		arhoFm = Uq[:, :, iarhoFm]
		rhoSlip  = Uq[:, :, irhoslip]

		# Extract momentum in vector form ([n, nq, ndims])
		mom    = Uq[:, :, smom]

		# Compute mixture (total) density
		rho = arhoA + arhoWv + arhoM
		p = self.compute_additional_variable("Pressure", Uq, True).squeeze(axis=2)
		u = rhou / rho
		v = rhov / rho
		u2 = u**2.0
		v2 = v**2.0
		rhouv = rhou * v
		# Vector
		vel = mom / np.expand_dims(rho,axis=2)

		# Construct physical flux
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		# Compute flux of non-tracer mass quantities in all directions
		F[:, :, self.get_mass_slice(),  :] = np.einsum(
			"ijk, ijl -> ijkl", Uq[:,:,self.get_mass_slice()], vel)
		F[:, :, irhou,   0] = rho * u2 + p        # x-flux of x-momentum
		F[:, :, irhov,   0] = rhouv               # x-flux of y-momentum
		F[:, :, irhou,   1] = rhouv               # y-flux of x-momentum
		F[:, :, irhov,   1] = rho * v2 + p        # y-flux of y-momentum
		F[:, :, ie,      :] = np.expand_dims(e + p,axis=2) * vel    # Flux of energy in all directions
		F[:, :, iarhoWt, :] = np.expand_dims(arhoWt,axis=2) * vel   # Flux of massWt in all directions
		F[:, :, iarhoC,  :] = np.expand_dims(arhoC,axis=2) * vel    # Flux of massC in all directions
		F[:, :, iarhoFm, :] = np.expand_dims(arhoFm,axis=2) * vel    # Flux of massFm in all directions
		F[:, :, irhoslip,  :] = np.expand_dims(rhoSlip,axis=2) * vel    # Flux of slip in all directions

		# Compute sound speed
		a = self.compute_additional_variable("SoundSpeed", Uq, True).squeeze(axis=2)

		return  F, (u2, v2, a)
