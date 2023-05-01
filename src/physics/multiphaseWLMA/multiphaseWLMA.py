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
#       File : src/physics/multiphaseWLMA/multiphasevpT.py
#
#       WLMA implementation in Quail. See also ./functions.py
#
# ------------------------------------------------------------------------ #
from enum import Enum
from logging import warning
import numpy as np

import errors
import general

# Inherited types
import physics.base
import physics.multiphasevpT.multiphasevpT as MultiphasevpT
import physics.multiphasevpT.functions as mpvpT_fcns
import physics.multiphaseWLMA.functions as mpWLMA_fcns

from physics.multiphaseWLMA.functions import FcnType
from physics.multiphaseWLMA.functions import SourceType
import physics.multiphaseWLMA.iapws95_light.mixtureWLMA as mixtureWLMA

class MultiphaseWLMA(MultiphasevpT.MultiphasevpT):
	'''
	This class inherits MultiphasevpT and replaces the pressure calculation.
	'''
	PHYSICS_TYPE = general.PhysicsType.MultiphaseWLMA # TODO: add to general. ...

	def __init__(self, mesh):
		super().__init__(mesh)

	def set_maps(self):
		# Set physics base maps, but skip parent set_maps
		physics.base.base.PhysicsBase.set_maps(self)

		self.BC_map.update({
			physics.base.functions.BCType.StateAll:
			  physics.base.functions.StateAll,
			physics.base.functions.BCType.Extrapolate:
			  physics.base.functions.Extrapolate,
			physics.multiphasevpT.functions.BCType.SlipWall:
			  physics.multiphasevpT.functions.SlipWall,
			physics.multiphaseWLMA.functions.BCType.LinearizedImpedance2D:
			  physics.multiphaseWLMA.functions.LinearizedImpedance2D,
			physics.multiphaseWLMA.functions.BCType.LinearizedImpedance2D:
				physics.multiphaseWLMA.functions.LinearizedImpedance2D,
			# BCType.MultiphasevpT2D2D: mpvpT_fcns.MultiphasevpT2D2D,
			# BCType.MultiphasevpT1D1D: mpvpT_fcns.MultiphasevpT1D1D,
			# BCType.MultiphasevpT2D1D: mpvpT_fcns.MultiphasevpT2D1D,
			# BCType.MultiphasevpT2D1DCylindrical: mpvpT_fcns.MultiphasevpT2D1DCylindrical,
			# BCType.MultiphasevpT2D2DCylindrical: mpvpT_fcns.MultiphasevpT2D2DCylindrical,
		})

	def set_physical_params(self, 
													Gas1={"R": 287., "gamma": 1.4},
													Gas2={"R": 8.314/18.02e-3, "c_p": 2.288e3}, 
													Liquid={"K": 10e9, "rho0": 2.6e3, "p0": 5e6,
																	"E_m0": 0, "c_m": 3e3},
													Solubility={"k": 5e-6, "n": 0.5},
													Viscosity={"mu0": 3e5},
													tau_d = 0.5):
		super().set_physical_params(Gas1=Gas1, Gas2=Gas2, Liquid=Liquid,
			      Solubility=Solubility, Viscosity=Viscosity, tau_d=tau_d)
		# Initialize WLMA object
		self.wlma = mixtureWLMA.WLMA(
			K=Liquid["K"],
			p_m0=Liquid["p0"],
			rho_m0=Liquid["rho0"],
			c_v_m0=Liquid["c_m"],
			R_a=Gas1["R"],
			gamma_a=Gas1["gamma"])
		self.Gas = [Gas1, Gas2]
		self.Liquid = Liquid
		self.Solubility = Solubility
		self.Viscosity = Viscosity
		self.tau_d = tau_d

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
		arhoC = Uq[:, :, self.get_state_slice("pDensityC")]
		arhoFm = Uq[:, :, self.get_state_slice("pDensityFm")]

		''' Flag non-physical state
		The EOS-constrained phases (A, Wv, M) are checked for positivity. Total
		water content, which is used in dissolution/exsolution, is also checked.
		'''
		if flag_non_physical:
			if np.any(arhoA < 0.) or np.any(arhoWv < 0.) or np.any(arhoM < 0.) \
				 or np.any(arhoWt < 0.): # or np.any(arhoC < 0.):
				raise errors.NotPhysicalError

		''' Nested functions for common quantities 
		Common routines that may be called for computing several outputs.
		States arhoA, mom, e, etc. are captured by nested functions at this point.
		'''
	
		''' Compute '''
		vname = self.AdditionalVariables[var_name].name

		if vname is self.AdditionalVariables["Pressure"].name:
			rhow, p, T, sound_speed, volfracW = \
				self.wlma(Uq[:,:,self.get_mass_slice()], mom, e)
			varq = p
		elif vname is self.AdditionalVariables["Temperature"].name:
			rhow, p, T, sound_speed, volfracW = \
				self.wlma(Uq[:,:,self.get_mass_slice()], mom, e)
			varq = T
		# elif vname is self.AdditionalVariables["Entropy"].name:
			# varq = R*(gamma/(gamma-1.)*np.log(getT()) - np.log(getP()))
			# Alternate way
			# varq = np.log(get_pressure()/rho**gamma)
		elif vname is self.AdditionalVariables["InternalEnergy"].name:
			varq = e - 0.5*np.sum(mom*mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["Enthalpy"].name:
			rhow, p, T, sound_speed, volfracW = \
				self.wlma(Uq[:,:,self.get_mass_slice()], mom, e)
			varq = e \
				- 0.5*np.sum(mom*mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM) \
				+ p / (arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["TotalEnthalpy"].name:
			rhow, p, T, sound_speed, volfracW = \
				self.wlma(Uq[:,:,self.get_mass_slice()], mom, e)
			varq = (e + p)/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["SoundSpeed"].name:
			rhow, p, T, sound_speed, volfracW = \
				self.wlma(Uq[:,:,self.get_mass_slice()], mom, e)
			varq = sound_speed
		elif vname is self.AdditionalVariables["MaxWaveSpeed"].name:
			# |u| + c
			rhow, p, T, sound_speed, volfracW = \
				self.wlma(Uq[:,:,self.get_mass_slice()], mom, e)
			varq = np.linalg.norm(mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM) \
				 + sound_speed
		elif vname is self.AdditionalVariables["Velocity"].name:
			varq = np.linalg.norm(mom, axis=2, keepdims=True)/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["XVelocity"].name:
			varq = mom[:, :, [0]]/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["YVelocity"].name:
			varq = mom[:, :, [1]]/(arhoA+arhoWv+arhoM)
		elif vname is self.AdditionalVariables["phi"].name:
			raise NotImplementedError
		elif vname is self.AdditionalVariables["volFracA"].name:
			raise NotImplementedError
			phi = get_porosity()
			# Compute gas partial pressures except where total gas mass is zero
			ppA = arhoA[np.where(phi > 0)] * self.Gas[0]["R"]
			ppWv = arhoWv[np.where(phi > 0)] * self.Gas[1]["R"]
			varq = phi
			varq[np.where(phi > 0)] = phi[np.where(phi > 0)] * (ppA / (ppA + ppWv))
		elif vname is self.AdditionalVariables["volFracWv"].name:
			rhow, p, T, sound_speed, volfracW = \
				self.wlma(Uq[:,:,self.get_mass_slice()], mom, e)
			varq = volfracW
			# phi = get_porosity()
			# # Compute gas partial pressures except where total gas mass is zero
			# ppA = arhoA[np.where(phi > 0)] * self.Gas[0]["R"]
			# ppWv = arhoWv[np.where(phi > 0)] * self.Gas[1]["R"]
			# varq = phi
			# varq[np.where(phi > 0)] = phi[np.where(phi > 0)] * (ppWv / (ppA + ppWv))
		elif vname is self.AdditionalVariables["volFracM"].name:
			raise NotImplementedError
			varq = 1.0 - get_porosity()
		elif vname is self.AdditionalVariables["Drag"].name:
			raise NotImplementedError
			varq = get_Drag()
		elif vname is self.AdditionalVariables["SpecificEntropy"].name:
			raise NotImplementedError
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
		 # TODO:
		raise NotImplementedError(f"pressure-xgradient not supported.")
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
		# TODO:
		raise NotImplementedError(f"pressure-sgradient not supported.")
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
		Compute the state-gradient of porosity phi.
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
		# TODO:
		raise NotImplementedError(f"phi-sgradient not supported.")


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

		# # Multiply with dU/dx for spatial gradient (function of grad_Uq)
		# dphidx = np.einsum('ijk, ijkl -> ijl', dphidU, grad_Uq)
		return dphidU

class MultiphaseWLMA2D(MultiphaseWLMA):
	'''
	WLMA model for 2D.

	Additional methods and attributes are commented below.
	'''
	NUM_STATE_VARS = 6+3 # (essential states + tracer states)
	NDIMS = 2

	def __init__(self, mesh):
		super().__init__(mesh)

	def set_maps(self):
		super().set_maps()

		d = {
			physics.multiphaseWLMA.functions.FcnType.IsothermalAtmosphere:
			  physics.multiphaseWLMA.functions.IsothermalAtmosphere,
		}

		self.IC_fcn_map.update(d)
		self.exact_fcn_map.update(d)
		self.BC_fcn_map.update(d)

		self.source_map.update({
			physics.multiphasevpT.functions.SourceType.GravitySource:
			  physics.multiphasevpT.functions.GravitySource,
			physics.multiphasevpT.functions.SourceType.CylindricalGeometricSource:
			  physics.multiphasevpT.functions.CylindricalGeometricSource,
		})

		self.conv_num_flux_map.update({
			physics.base.functions.ConvNumFluxType.LaxFriedrichs:
				physics.multiphasevpT.functions.LaxFriedrichs2D,
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
		return iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm

	def get_mass_slice(self):
		''' Get slice representation of mass variables for each phase, such that
		the slice returned sums to the mixture density along axis=-1. '''
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
		iarhoA, iarhoWv, iarhoM, irhou, irhov, ie, iarhoWt, iarhoC, iarhoFm = \
			self.get_state_indices()
		# Extract data of size [n, nq]
		arhoA  = Uq[:, :, 0:1]
		arhoWv = Uq[:, :, 1:2]
		arhoM  = Uq[:, :, 2:3]
		rhou   = Uq[:, :, 3:4]
		rhov   = Uq[:, :, 4:5]
		e      = Uq[:, :, 5:6]
		arhoWt = Uq[:, :, 6:7]
		arhoC  = Uq[:, :, 7:8]
		arhoFm = Uq[:, :, 8:9]

		# Extract momentum in vector form ([n, nq, ndims])
		mom    = Uq[:, :, self.get_momentum_slice()]

		# Call mixture backend
		rhow, p, T, sound_speed, volfracW = \
			self.wlma(
				Uq[:,:,self.get_mass_slice()],
				mom,
				Uq[:, :, self.get_state_slice("Energy")])
		# Compute mixture (total) density
		rho = arhoA + arhoWv + arhoM
		u = rhou / rho
		v = rhov / rho
		u2 = u**2.0
		v2 = v**2.0
		rhouv = rhou * v
		# Vector
		vel = mom / rho

		# Construct physical flux
		F = np.empty(Uq.shape + (self.NDIMS,)) # [n, nq, ns, ndims]
		# Compute flux of non-tracer mass quantities in all directions
		F[:, :, self.get_mass_slice(),  :] = np.einsum(
			"ijk, ijl -> ijkl", Uq[:,:,self.get_mass_slice()], vel)
		F[:, :, irhou,   0:1] = rho * u2 + p        # x-flux of x-momentum
		F[:, :, irhov,   0:1] = rhouv               # x-flux of y-momentum
		F[:, :, irhou,   1:2] = rhouv               # y-flux of x-momentum
		F[:, :, irhov,   1:2] = rho * v2 + p        # y-flux of y-momentum
		F[:, :, ie,      :] = (e + p) * vel    # Flux of energy in all directions
		F[:, :, iarhoWt, :] = arhoWt * vel   # Flux of massWt in all directions
		F[:, :, iarhoC,  :] = arhoC * vel    # Flux of massC in all directions
		F[:, :, iarhoFm, :] = arhoFm * vel    # Flux of massFm in all directions

		return  F, (u2, v2, sound_speed)
