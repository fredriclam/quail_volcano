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
#       File : src/numerics/limiting/positivitypreserving.py
#
#       Contains class definitions for positivity-preserving limiters.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

import errors
import general

import meshing.tools as mesh_tools

import numerics.helpers.helpers as helpers
import numerics.limiting.base as base

import pickle

POS_TOL = 1e-6 # 1.e-10


def trunc(a, decimals=8):
	'''
	This function truncates a float to a specified decimal place.
	Adapted from:
	https://stackoverflow.com/questions/42021972/
	truncating-decimal-digits-numpy-array-of-floats

	Inputs:
	-------
		a: value(s) to truncate
		decimals: truncated decimal place

	Outputs:
	--------
		truncated float
	'''
	return np.trunc(a*10**decimals)/(10**decimals)


class PositivityPreserving(base.LimiterBase):
	'''
	This class corresponds to the positivity-preserving limiter for the
	Euler equations. It inherits from the LimiterBase class. See
	See LimiterBase for detailed comments of attributes and methods. See
	the following references:
		[1] X. Zhang, C.-W. Shu, "On positivity-preserving high order
		discontinuous Galerkin schemes for compressible Euler equations
		on rectangular meshes," Journal of Computational Physics.
		229:8918–8934, 2010.
		[2] C. Wang, X. Zhang, C.-W. Shu, J. Ning, "Robust high order
		discontinuous Galerkin schemes for two-dimensional gaseous
		detonations," Journal of Computational Physics, 231:653-665, 2012.

	Attributes:
	-----------
	var_name1: str
		name of first variable involved in limiting (density)
	var_name2: str
		name of second variable involved in limiting (pressure)
	elem_vols: numpy array
		element volumes
	basis_val_elem_faces: numpy array
		stores basis values for element and faces
	quad_wts_elem: numpy array
		quadrature points for element
	djac_elems: numpy array
		stores Jacobian determinants for each element
	'''
	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Euler

	def __init__(self, physics_type):
		super().__init__(physics_type)
		self.var_name1 = "Density"
		self.var_name2 = "Pressure"
		self.elem_vols = np.zeros(0)
		self.basis_val_elem_faces = np.zeros(0)
		self.quad_wts_elem = np.zeros(0)
		self.djac_elems = np.zeros(0)

	def precompute_helpers(self, solver):
		# Unpack
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		self.elem_vols, _ = mesh_tools.element_volumes(solver.mesh, solver)

		# Basis values in element interior and on faces
		if not solver.basis.skip_interp:
			basis_val_faces = int_face_helpers.faces_to_basisL.copy()
			bshape = basis_val_faces.shape
			basis_val_faces.shape = (bshape[0]*bshape[1], bshape[2])
			self.basis_val_elem_faces = np.vstack((elem_helpers.basis_val,
					basis_val_faces))
		else:
			self.basis_val_elem_faces = elem_helpers.basis_val

		# Jacobian determinant
		self.djac_elems = elem_helpers.djac_elems

		# Element quadrature weights
		self.quad_wts_elem = elem_helpers.quad_wts

	def limit_solution(self, solver, Uc):
		# Unpack
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		basis = solver.basis

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]
		ne = self.elem_vols.shape[0]
		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)

		# Density and pressure from averaged state
		rho_bar = physics.compute_variable(self.var_name1, U_bar)
		p_bar = physics.compute_variable(self.var_name2, U_bar)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.):
			raise errors.NotPhysicalError

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density at quadrature points
		rho_elem_faces = physics.compute_variable(self.var_name1,
				U_elem_faces)
		# Check if limiting is needed
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		# Truncate theta1; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta1 = trunc(np.minimum(1., np.min(theta, axis=1)))

		irho = physics.get_state_index(self.var_name1)
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta1 < 1.)[0]
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs, :, irho] = theta1[elem_IDs]*Uc[elem_IDs, :, irho] \
					+ (1. - theta1[elem_IDs])*rho_bar[elem_IDs, 0]
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs, :, irho] *= theta1[elem_IDs]
			Uc[elem_IDs, 0, irho] += (1. - theta1[elem_IDs, 0])*rho_bar[
					elem_IDs, 0, 0]
		else:
			raise NotImplementedError

		if np.any(theta1 < 1.):
			# Intermediate limited solution
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)

		''' Limit pressure '''
		# Compute pressure at quadrature points
		p_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces)
		theta[:] = 1.
		# Indices where pressure is negative
		negative_p_indices = np.where(p_elem_faces < 0.)
		elem_IDs = negative_p_indices[0]
		i_neg_p  = negative_p_indices[1]

		theta[elem_IDs, i_neg_p] = (p_bar[elem_IDs, :, 0] - POS_TOL) / (
				p_bar[elem_IDs, :, 0] - p_elem_faces[elem_IDs, i_neg_p, :])

		# Truncate theta2; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta2 = trunc(np.min(theta, axis=1))
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta2 < 1.)[0]
		# Modify coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs] = np.einsum('im, ijk -> ijk', theta2[elem_IDs], 
					Uc[elem_IDs]) + np.einsum('im, ijk -> ijk', 1 - theta2[
					elem_IDs], U_bar[elem_IDs])
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs] *= np.expand_dims(theta2[elem_IDs], axis=2)
			Uc[elem_IDs, 0] += np.einsum('im, ijk -> ik', 1 - theta2[
					elem_IDs], U_bar[elem_IDs])
		else:
			raise NotImplementedError

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]


class PositivityPreservingChem(PositivityPreserving):
	'''
    Class: PPLimiter
    ------------------
    This class contains information about the positivity preserving limiter
    '''

	COMPATIBLE_PHYSICS_TYPES = general.PhysicsType.Chemistry

	def __init__(self, physics_type):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		super().__init__(physics_type)
		self.var_name3 = "Mixture"


	def limit_solution(self, solver, Uc):
		# Unpack
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		basis = solver.basis

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]

		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)
		ne = self.elem_vols.shape[0]
		# Density and pressure from averaged state
		rho_bar = physics.compute_variable(self.var_name1, U_bar)
		p_bar = physics.compute_variable(self.var_name2, U_bar)
		rhoY_bar = physics.compute_variable(self.var_name3, U_bar)

		if np.any(rho_bar < 0.) or np.any(p_bar < 0.) or np.any(
				rhoY_bar < 0.):
			raise errors.NotPhysicalError

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		''' Limit density '''
		# Compute density
		rho_elem_faces = physics.compute_variable(self.var_name1,
				U_elem_faces)
		# Check if limiting is needed
		theta = np.abs((rho_bar - POS_TOL)/(rho_bar - rho_elem_faces))
		# Truncate theta1; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta1 = trunc(np.minimum(1., np.min(theta, axis=1)))

		irho = physics.get_state_index(self.var_name1)
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta1 < 1.)[0]
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs, :, irho] = theta1[elem_IDs]*Uc[elem_IDs, :, irho] \
					+ (1. - theta1[elem_IDs])*rho_bar[elem_IDs, 0]
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs, :, irho] *= theta1[elem_IDs]
			Uc[elem_IDs, 0, irho] += (1. - theta1[elem_IDs, 0])*rho_bar[
					elem_IDs, 0, 0]
		else:
			raise NotImplementedError

		if np.any(theta1 < 1.):
			# Intermediate limited solution
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)


		''' Limit mass fraction '''
		rhoY_elem_faces = physics.compute_variable(self.var_name3, U_elem_faces)
		theta = np.abs(rhoY_bar/(rhoY_bar-rhoY_elem_faces+POS_TOL))
		# Truncate theta2; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta2 = trunc(np.minimum(1., np.amin(theta, axis=1)))

		irhoY = physics.get_state_index(self.var_name3)
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta2 < 1.)[0]
		# Modify density coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs, :, irhoY] = theta2[elem_IDs]*Uc[elem_IDs, :, 
					irhoY] + (1. - theta2[elem_IDs])*rho_bar[elem_IDs, 0]
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs, :, irhoY] *= theta2[elem_IDs]
			Uc[elem_IDs, 0, irhoY] += (1. - theta2[elem_IDs, 0])*rho_bar[
					elem_IDs, 0, 0]
		else:
			raise NotImplementedError

		if np.any(theta2 < 1.):
			U_elem_faces = helpers.evaluate_state(Uc,
					self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)

		''' Limit pressure '''
		# Compute pressure at quadrature points
		p_elem_faces = physics.compute_variable(self.var_name2, U_elem_faces)
		theta[:] = 1.
		# Indices where pressure is negative
		negative_p_indices = np.where(p_elem_faces < 0.)
		elem_IDs = negative_p_indices[0]
		i_neg_p  = negative_p_indices[1]

		theta[elem_IDs, i_neg_p] = p_bar[elem_IDs, :, 0] / (
				p_bar[elem_IDs, :, 0] - p_elem_faces[elem_IDs, i_neg_p])

		# Truncate theta3; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta3 = trunc(np.min(theta, axis=1))
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta3 < 1.)[0]
		# Modify coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs] = np.einsum('im, ijk -> ijk', theta3[elem_IDs], 
					Uc[elem_IDs]) + np.einsum('im, ijk -> ijk', 1 - theta3[
					elem_IDs], U_bar[elem_IDs])
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs] *= np.expand_dims(theta3[elem_IDs], axis=2)
			Uc[elem_IDs, 0] += np.einsum('im, ijk -> ik', 1 - theta3[
					elem_IDs], U_bar[elem_IDs])
		else:
			raise NotImplementedError

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]


class PositivityPreservingMultiphasevpT(PositivityPreserving):
	'''
    Class: PPLimiter
    ------------------
    This class contains information about the positivity preserving limiter
    '''

	COMPATIBLE_PHYSICS_TYPES = [general.PhysicsType.MultiphasevpT,
			     general.PhysicsType.MultiphaseWLMA]

	def __init__(self, physics_type):
		'''
		Method: __init__
		-------------------
		Initializes PPLimiter object
		'''
		super().__init__(physics_type)
		self.var_name1 = "pDensityA"
		self.var_name2 = "pDensityWv"
		self.var_name3 = "pDensityM"
		self.var_name4 = "Pressure"
		# Tracer variables
		self.var_nameT1 = "pDensityWt"
		self.var_nameT2 = "pDensityC"

		# self.theta_store = [[], [], [], []]
	
	# def __del__(self):
	# 	print("Dismantling logging in positivitypreserving.py/PositivityPreservingMultiphasevpT")
	# 	with open('log_data.dat', 'wb') as f:
	# 		pickle.dump(self.theta_store, f)

	def limit_solution_legacy(self, solver, Uc):
		# Unpack
		physics = solver.physics
		elem_helpers = solver.elem_helpers
		int_face_helpers = solver.int_face_helpers
		basis = solver.basis

		djac = self.djac_elems

		# Interpolate state at quadrature points over element and on faces
		U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
				skip_interp=basis.skip_interp)
		nq_elem = self.quad_wts_elem.shape[0]
		U_elem = U_elem_faces[:, :nq_elem, :]

		# Average value of state
		U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
				self.elem_vols)
		ne = self.elem_vols.shape[0]
		# Density and pressure from averaged state
		rho_bar = {self.var_name1: physics.compute_variable(self.var_name1, U_bar),
		           self.var_name2: physics.compute_variable(self.var_name2, U_bar),
							 self.var_name3: physics.compute_variable(self.var_name3, U_bar),
							 self.var_nameT1: physics.compute_variable(self.var_nameT1, U_bar),
							 self.var_nameT2: physics.compute_variable(self.var_nameT2, U_bar),
							 }
		p_bar = physics.compute_variable(self.var_name4, U_bar)

		if np.any([np.any(rho < 0.) for rho in rho_bar.values()]) \
			 or np.any(p_bar < 0.):
			raise errors.NotPhysicalError

		# Ignore divide-by-zero
		np.seterr(divide='ignore')

		# logger_idx = -1
		''' Limit partial-density variables '''
		for var_name in [self.var_name1, self.var_name2, self.var_name3,
			self.var_nameT1, self.var_nameT2]:
			# logger_idx += 1
			# Compute density
			quant_elem_faces = physics.compute_variable(var_name, U_elem_faces)
			# Evaluate theta parameter
			if self.var_name3 == var_name:
				# Liquid phase: separate theta evaluation
				theta = np.abs((rho_bar[var_name])/(
					rho_bar[var_name] - quant_elem_faces + POS_TOL))
			else:
				denom = np.where(rho_bar[var_name] - quant_elem_faces != 0,
					rho_bar[var_name] - quant_elem_faces, POS_TOL)
				theta = np.abs((rho_bar[var_name] - POS_TOL)/denom)
			# theta = np.abs((rho_bar[var_name])/(
			# 	rho_bar[var_name] - quant_elem_faces + POS_TOL))
			# Truncate theta1; otherwise, can get noticeably different
			# results across machines, possibly due to poor conditioning in its
			# calculation
			theta1 = trunc(np.minimum(1., np.min(theta, axis=1)))
			# self.theta_store[logger_idx].append(theta1)

			# Get component's partial density index
			irho = physics.get_state_index(var_name)
			# Get IDs of elements that need limiting
			elem_IDs = np.where(theta1 < 1.)[0]
			# Modify density coefficients
			if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
				Uc[elem_IDs, :, irho] = theta1[elem_IDs]*Uc[elem_IDs, :, irho] \
						+ (1. - theta1[elem_IDs])*rho_bar[var_name][elem_IDs, 0]
			elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
				Uc[elem_IDs, :, irho] *= theta1[elem_IDs]
				Uc[elem_IDs, 0, irho] += (1. - theta1[elem_IDs, 0])*rho_bar[var_name][
						elem_IDs, 0, 0]
			else:
				raise NotImplementedError

			if np.any(theta1 < 1.):
				# Intermediate limited solution
				U_elem_faces = helpers.evaluate_state(Uc,
						self.basis_val_elem_faces,
						skip_interp=basis.skip_interp)


		# ''' Limit mass fraction '''
		# rhoY_elem_faces = physics.compute_variable(self.var_name3, U_elem_faces)
		# theta = np.abs(rhoY_bar/(rhoY_bar-rhoY_elem_faces+POS_TOL))
		# # Truncate theta2; otherwise, can get noticeably different
		# # results across machines, possibly due to poor conditioning in its
		# # calculation
		# theta2 = trunc(np.minimum(1., np.amin(theta, axis=1)))

		# irhoY = physics.get_state_index(self.var_name3)
		# # Get IDs of elements that need limiting
		# elem_IDs = np.where(theta2 < 1.)[0]
		# # Modify density coefficients
		# if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
		# 	Uc[elem_IDs, :, irhoY] = theta2[elem_IDs]*Uc[elem_IDs, :, 
		# 			irhoY] + (1. - theta2[elem_IDs])*rho_bar[elem_IDs, 0]
		# elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
		# 	Uc[elem_IDs, :, irhoY] *= theta2[elem_IDs]
		# 	Uc[elem_IDs, 0, irhoY] += (1. - theta2[elem_IDs, 0])*rho_bar[
		# 			elem_IDs, 0, 0]
		# else:
		# 	raise NotImplementedError

		# if np.any(theta2 < 1.):
		# 	U_elem_faces = helpers.evaluate_state(Uc,
		# 			self.basis_val_elem_faces,
		# 			skip_interp=basis.skip_interp)

		''' Limit pressure '''
		# Compute pressure at quadrature points
		p_elem_faces = physics.compute_variable(self.var_name4, U_elem_faces)
		theta[:] = 1.
		# Indices where pressure is negative
		negative_p_indices = np.where(p_elem_faces < 0.)
		elem_IDs = negative_p_indices[0]
		i_neg_p  = negative_p_indices[1]

		# theta[elem_IDs, i_neg_p] = p_bar[elem_IDs, :, 0] / (
		# 		p_bar[elem_IDs, :, 0] - p_elem_faces[elem_IDs, i_neg_p])
		# Modification for low-pressure Noh usage:
		theta[elem_IDs, i_neg_p] = np.clip((p_bar[elem_IDs, :, 0] - 1e-5) / (
				p_bar[elem_IDs, :, 0] - p_elem_faces[elem_IDs, i_neg_p]), 0, 1)

		# Truncate theta3; otherwise, can get noticeably different
		# results across machines, possibly due to poor conditioning in its
		# calculation
		theta3 = trunc(np.min(theta, axis=1))
		# self.theta_store[3].append(theta3)
		# Get IDs of elements that need limiting
		elem_IDs = np.where(theta3 < 1.)[0]
		# Modify coefficients
		if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
			Uc[elem_IDs] = np.einsum('im, ijk -> ijk', theta3[elem_IDs], 
					Uc[elem_IDs]) + np.einsum('im, ijk -> ijk', 1 - theta3[
					elem_IDs], U_bar[elem_IDs])
		elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
			Uc[elem_IDs] *= np.expand_dims(theta3[elem_IDs], axis=2)
			Uc[elem_IDs, 0] += np.einsum('im, ijk -> ik', 1 - theta3[
					elem_IDs], U_bar[elem_IDs])
		else:
			raise NotImplementedError

		np.seterr(divide='warn')

		return Uc # [ne, nq, ns]
	
	def limit_solution(self, solver, Uc):
			# Unpack
			physics = solver.physics
			elem_helpers = solver.elem_helpers
			int_face_helpers = solver.int_face_helpers
			basis = solver.basis

			djac = self.djac_elems

			# Interpolate state at quadrature points over element and on faces
			U_elem_faces = helpers.evaluate_state(Uc, self.basis_val_elem_faces,
					skip_interp=basis.skip_interp)
			

			# Compute cell-average state
			nq_elem = self.quad_wts_elem.shape[0]
			ne = self.elem_vols.shape[0]
			U_elem = U_elem_faces[:, :nq_elem, :]
			U_bar = helpers.get_element_mean(U_elem, self.quad_wts_elem, djac,
					self.elem_vols)
			
			# Density and pressure from averaged state
			if physics.PHYSICS_TYPE == general.PhysicsType.MultiphaseWLMA:
				rho_bar = {self.var_name1: physics.compute_variable(self.var_name1, U_bar),
								self.var_name2: physics.compute_variable(self.var_name2, U_bar),
								self.var_name3: physics.compute_variable(self.var_name3, U_bar),
								}
			else:
				rho_bar = {self.var_name1: physics.compute_variable(self.var_name1, U_bar),
									self.var_name2: physics.compute_variable(self.var_name2, U_bar),
									self.var_name3: physics.compute_variable(self.var_name3, U_bar),
									self.var_nameT1: physics.compute_variable(self.var_nameT1, U_bar),
									# self.var_nameT2: physics.compute_variable(self.var_nameT2, U_bar),
									}
			
			p_bar = physics.compute_variable("Pressure", U_bar)

			if np.any([np.any(rho < 0.) for rho in rho_bar.values()]) \
				or np.any(p_bar < 0.):
				# Unrecoverable if pressure or density based on cell-average is negative
				raise errors.NotPhysicalError

			# Ignore divide-by-zero
			np.seterr(divide='ignore')

			''' Theta modified here following Zhang, Xia, Shu (J. Sci. Comp. 2012 ),
			where m = POS_TOL and M = +inf. '''

			def quadratic_extrema(Uc) -> np.array:
				# Analytic quadratic program (QP) solution for element minimum of each
				# state variable, independently. Note that the resulting
				# candidate_states are not state vectors at particular points; these
				# contain the extreme values of the state scalars independently. 
				# Compute first-order condition == linear system coefficients
				A00 = 4*Uc[:,2,:] + 4*Uc[:,0,:] - 8*Uc[:,1,:]
				A01 = 4*Uc[:,0,:] - 4*Uc[:,3,:] - 4*Uc[:,1,:] + 4*Uc[:,4,:]
				A10 = A01
				A11 = 4*Uc[:,5,:] + 4*Uc[:,0,:] - 8*Uc[:,3,:]
				b0 = Uc[:,2,:] + 3*Uc[:,0,:] - 4*Uc[:,1,:]
				b1 = Uc[:,5,:] + 3*Uc[:,0,:] - 4*Uc[:,3,:]
				# y = 0 edge
				A_y0 = A00
				b_y0 = b0
				# x = 0 edge
				A_x0 = A11
				b_x0 = b1
				# t = 0 edge
				A_t0 = 4*Uc[:,2,:] - 8*Uc[:,4,:] + 4*Uc[:,5,:]
				b_t0 = Uc[:,2,:] - 4*Uc[:,4,:] + 3*Uc[:,5,:]
				# Allocate zeros for candidates points
				#   with shape (n_elements, n_states, n_candidates=7, n_dims==2)
				candidate_points = np.zeros((Uc.shape[0], Uc.shape[2], 7, 2))
				# Compute 2D interior critical point
				det = A00 * A11 - A01 * A10
				np.divide(np.stack([A11 * b0 - A01 * b1, -A10 * b0 + A00 * b1], axis=-1),
									det[...,np.newaxis],
									out=candidate_points[:,:,0,:],
									where=(~np.isclose(det[...,np.newaxis], 0)))
				# Compute 1D boundary critical points
				np.divide(np.stack([b_y0, 0*b_y0], axis=-1),
									A_y0[...,np.newaxis],
									out=candidate_points[:,:,1,:],
									where=(~np.isclose(A_y0[...,np.newaxis], 0)))
				np.divide(np.stack([0*b_x0, b_x0], axis=-1),
									A_x0[...,np.newaxis],
									out=candidate_points[:,:,2,:],
									where=(~np.isclose(A_x0[...,np.newaxis], 0)))
				np.divide(np.stack([b_t0, A_t0 - b_t0], axis=-1),
									A_t0[...,np.newaxis],
									out=candidate_points[:,:,3,:],
									where=(~np.isclose(A_t0[...,np.newaxis], 0)))
				# Include corner nodes
				candidate_points[:,:,4,:] = np.array([0, 0])
				candidate_points[:,:,5,:] = np.array([1, 0])
				candidate_points[:,:,6,:] = np.array([0, 1])
				# Restrict points to the triangle x >= 1, y >= 1, t = 1 - x - y >= 0
				candidate_points = np.clip(candidate_points, 0, 1)
				candidate_points[:,:,:,1] = np.minimum(candidate_points[:,:,:,1],
																					     1 - candidate_points[:,:,:,0])

				def eval_basis_at(x, coeffs):
					''' Evaluate basis against coefficients for arbitrary x
								x: shape (ne, ns, ncandidates==7, ndims=2)
								coeffs: shape (ne, nb==6, ns)
					'''
					_x = x[...,0]
					_y = x[...,1]
					_t = (1 - _x - _y)
					# Multiply coefficients against coordinates, newaxis for every sample point (last axis of x)
					return (coeffs[...,0,:,np.newaxis] * (_t * (2 * _t - 1))
								+ coeffs[...,1,:,np.newaxis] * (4 * _x * _t)
								+ coeffs[...,2,:,np.newaxis] * (_x * (2 * _x - 1))
								+ coeffs[...,3,:,np.newaxis] * (4 * _y * _t)
								+ coeffs[...,4,:,np.newaxis] * (4 * _x * _y)
								+ coeffs[...,5,:,np.newaxis] * (_y * (2 * _y - 1)))
				# Compute candidate_states (ne, ns, 8) and swap to (ne, 8, ns)
				candidate_states = np.swapaxes(eval_basis_at(candidate_points, Uc), 1, 2)
				# Compute elementwise extrema(n_elements, 1, n_states)
				# elt_max = basis_vec(candidate_points, Uc).max(axis=2)[:,np.newaxis,:]

				return candidate_states


			if solver.order == 0 or solver.order == 1 or solver.physics.NDIMS == 1:
				# Concatenate face/element quadrature point values and nodal points
				candidate_states = np.concatenate((U_elem_faces, Uc), axis=1)
				elt_min = candidate_states.min(axis=1, keepdims=True)
			elif solver.order == 2:
				# Append quadrature point states and nodal point states
				candidate_states = np.concatenate((quadratic_extrema(Uc), U_elem_faces, Uc),axis=1 )
				elt_min = candidate_states.min(axis=1, keepdims=True)
			else:
				assert solver.order <= 2, "For higher order, implement piecewise polynomial extreme check for PPL"
			
			# Select non-momentum indices for limiting
			if solver.physics.NDIMS == 1:
				_i = [0,1,2,4,5,6,7,]
			elif solver.physics.NDIMS == 2:
				_i = [0,1,2,5,6,7,8,]
			denom = np.where(U_bar[...,_i] - elt_min[...,_i] != 0,
											U_bar[...,_i] - elt_min[...,_i], POS_TOL)
			pos_tol_vec = POS_TOL * np.ones_like(denom)
			pos_tol_vec[...,3] *= 1e5 # Energy scaling
			theta = np.abs((U_bar[...,_i] - pos_tol_vec)/denom).min(axis=2, keepdims=True)
					
			# Pick strictest theta
			theta1 = trunc(np.minimum(1., np.min(theta, axis=1, keepdims=True)))

			# Get IDs of elements that need limiting
			elem_IDs = np.where(theta1 < 1.)[0]
			# Modify density coefficients
			if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
				Uc[elem_IDs, :, :] = theta1[elem_IDs] * Uc[elem_IDs, :, :] \
					+ (1.0 - theta1[elem_IDs]) * U_bar[elem_IDs, :, :]
				# Uc[elem_IDs, :, 0:3] = theta1[elem_IDs] * Uc[elem_IDs, :, 0:3] \
				# 	+ (1.0 - theta1[elem_IDs]) * mix_rho_bar[elem_IDs,...] * y[elem_IDs,...]
				# Uc[elem_IDs, :, 5:9] = theta1[elem_IDs] * Uc[elem_IDs, :, 5:9] \
				# 	+ (1.0 - theta1[elem_IDs]) * U_bar[elem_IDs, :, 5:9]
			elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
				raise NotImplementedError("Mod for modal in vpT not implemented.")
				irho = physics.get_state_index(var_name)
				Uc[elem_IDs, :, irho] *= theta1[elem_IDs]
				Uc[elem_IDs, 0, irho] += (1. - theta1[elem_IDs, 0])*rho_bar[var_name][
						elem_IDs, 0, 0]
			else:
				raise NotImplementedError

			if np.any(theta1 < 1.):
				# Re-evaluate intermediate limited solution
				U_elem_faces = helpers.evaluate_state(Uc,
						self.basis_val_elem_faces,
						skip_interp=basis.skip_interp)
				# Re-evaluate extrema states
				if solver.order == 0 or solver.order == 1 or solver.physics.NDIMS == 1:
					# Concatenate face/element quadrature point values and nodal points
					candidate_states = np.concatenate((U_elem_faces, Uc), axis=1)
					elt_min = candidate_states.min(axis=1, keepdims=True)
				elif solver.order == 2:
					# Append quadrature point states and nodal point states
					candidate_states = np.concatenate((quadratic_extrema(Uc), U_elem_faces, Uc),axis=1 )
					elt_min = candidate_states.min(axis=1, keepdims=True)
				else:
					assert solver.order <= 2, "For higher order, implement piecewise polynomial extreme check for PPL"

			''' Limit energy (for ideal gas, p is proportional to vol. internal energy) '''
			if physics.PHYSICS_TYPE == general.PhysicsType.MultiphaseWLMA:
				e_bar = U_bar[..., 5:6]
				if np.any(e_bar < 0.):
					raise errors.NotPhysicalError

				# Compute minimum within element (works for order == 1 using nodal vals)
				assert(solver.order <= 1)
				elt_min = Uc[..., 5:6].min(axis=1, keepdims=True)
				denom = np.where(e_bar - elt_min != 0, e_bar - elt_min, POS_TOL)
				theta = np.abs((e_bar - POS_TOL)/denom)

				# theta = np.abs((rho_bar[var_name])/(
				# 	rho_bar[var_name] - quant_elem_faces + POS_TOL))
				# Truncate theta1; otherwise, can get noticeably different
				# results across machines, possibly due to poor conditioning in its
				# calculation
				theta1 = trunc(np.minimum(1., np.min(theta, axis=1, keepdims=True)))
				# theta_test = (mix_rho_bar[...] - POS_TOL)/(mix_rho_bar[...] - Uc.sum(axis=-1, keepdims=True).min(axis=1, keepdims=True))

				# self.theta_store[logger_idx].append(theta1)

				# Get IDs of elements that need limiting
				elem_IDs = np.where(theta1 < 1.)[0]
				# Modify density coefficients
				if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
					Uc[elem_IDs, :, 5:6] = theta1[elem_IDs] * Uc[elem_IDs, :, 5:6] \
						+ (1.0 - theta1[elem_IDs]) * e_bar[elem_IDs,...]
				elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
					raise NotImplementedError("Mod for modal in vpT not implemented.")
				else:
					raise NotImplementedError

				if np.any(theta1 < 1.):
					# Intermediate limited solution
					U_elem_faces = helpers.evaluate_state(Uc,
							self.basis_val_elem_faces,
							skip_interp=basis.skip_interp)

			''' Limit pressure '''

			PRESSURE_TOL = 1e5 * POS_TOL

			estimate_with_quad_points = False
			if estimate_with_quad_points:
				# Legacy method
				# Compute pressure at quadrature points (interior and face)
				p_elem_faces = physics.compute_variable("Pressure", U_elem_faces)
				# Indices where pressure is negative
				theta = np.where(p_elem_faces < 0., (p_bar - POS_TOL) / (p_bar - p_elem_faces), 1.0)
			else:
				# Compute pressure at quadrature points
				p_candidates = physics.compute_variable("Pressure", candidate_states)
				# Indices where pressure is negative
				theta = np.clip(np.abs((p_bar - PRESSURE_TOL) / (p_bar - p_candidates)), 0, 1)
			
			theta3 = trunc(np.min(theta, axis=1))
			# Get IDs of elements that need limiting
			elem_IDs = np.where(theta3 < 1.)[0]
			# Modify coefficients
			if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
				Uc[elem_IDs] = np.einsum('im, ijk -> ijk', theta3[elem_IDs], 
						Uc[elem_IDs]) + np.einsum('im, ijk -> ijk', 1 - theta3[
						elem_IDs], U_bar[elem_IDs])
			elif basis.MODAL_OR_NODAL == general.ModalOrNodal.Modal:
				Uc[elem_IDs] *= np.expand_dims(theta3[elem_IDs], axis=2)
				Uc[elem_IDs, 0] += np.einsum('im, ijk -> ik', 1 - theta3[
						elem_IDs], U_bar[elem_IDs])
			else:
				raise NotImplementedError
			
			# Severe pressure limiting
			# Since pressure is nonlinear in the state variables, there is no
			# guarantee that a theta-weighted average is sufficient.
			theta_final = np.ones_like(theta3)
			if np.any(theta3 < 1.):
				# Re-evaluate intermediate limited solution
				U_elem_faces = helpers.evaluate_state(Uc,
						self.basis_val_elem_faces,
						skip_interp=basis.skip_interp)
				# Re-evaluate extrema states
				if solver.order == 0 or solver.order == 1 or solver.physics.NDIMS == 1:
					# Concatenate face/element quadrature point values and nodal points
					candidate_states = np.concatenate((U_elem_faces, Uc), axis=1)
					elt_min = candidate_states.min(axis=1, keepdims=True)
				elif solver.order == 2:
					# Append quadrature point states and nodal point states
					candidate_states = np.concatenate((quadratic_extrema(Uc), U_elem_faces, Uc),axis=1 )
					elt_min = candidate_states.min(axis=1, keepdims=True)
				else:
					assert solver.order <= 2, "For higher order, implement piecewise polynomial extreme check for PPL"
				# Compute pressure at quadrature points
				p_candidates = physics.compute_variable("Pressure", candidate_states)
				# Indices where pressure is negative
				theta_final = np.clip(np.abs((p_bar - PRESSURE_TOL) / (p_bar - p_candidates)), 0, 1)

			if np.any(theta_final < 1):
				# Get IDs of elements that need limiting
				elem_IDs = np.where(theta_final < 1.)[0]
				# Emergency shift to cell-average value
				if basis.MODAL_OR_NODAL == general.ModalOrNodal.Nodal:
					Uc[elem_IDs] = U_bar[elem_IDs]
				else:
					raise NotImplementedError

			np.seterr(divide='warn')

			# Postcheck
			if np.any(Uc[...,0:3] < 0.):
				raise errors.NotPhysicalError("An essential mass variable is negative")
			if np.any(Uc[...,5:6] < 0.) and solver.physics.NDIMS == 2:
				# Energy 
				raise errors.NotPhysicalError("Total energy is negative")

			return Uc # [ne, nq, ns]