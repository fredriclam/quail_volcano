# ------------------------------------------------------------------------ #
#
#       File : src/numerics/timestepping/source_stepper.py
#
#       Contains solvers for operator splitting schemes in stepper.py
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np
import scipy
from scipy.integrate import ode

import numerics.helpers.helpers as helpers

from solver.tools import mult_inv_mass_matrix
import solver.tools as solver_tools


class SourceSolvers():
	'''
	SourceSolvers is a class of classes used as a Mixin class for operator
	splitting approaches (see stepper.py). These methods are specifically
	used for solving ODEs of the form:
		dU/dt = S(U)
	Current schemes supported include:
		- Backward Difference (BDF1)
		- Trapezoidal Scheme (Trapezoidal)
		- Scipy's Stiff LSODA Scheme (LSODA)
	'''
	class SourceStepperBase(ABC):
		'''
		This is an abstract base class used to represent a specific ODE
		solver for operator splitting integration schemes.
		Attributes:
		-----------
		res: numpy array of floats (shape : [nelem, nb, ns])
			solution's residaul array
		dt: float
			time-step for the solution
		num_time_steps: int
			number of time steps for the given solution's FinalTime
		get_time_step: method
			method to obtain dt given input decks logic (CFL-based vs # of
			timesteps, etc...)
		balance_const: numpy array of floats (shaped like res)
			balancing constant array used only with the Simpler splitting
			scheme

		Abstract Methods:
		-----------------
		take_time_step
			method that takes a given time step for the solver depending on
			the selected time-stepping scheme
		'''
		def __init__(self, U):
			self.res = np.zeros_like(U)
			self.dt = 0.
			self.num_time_steps = 0
			self.get_time_step = None
			self.balance_const = None

		def __repr__(self):
			return '{self.__class__.__name__}(TimeStep={self.dt})'.format( \
					self=self)

	class BDF1(SourceStepperBase):
		'''
		Re-implementation of 1st-order Backward Differencing (BDF1) aka
		backward Euler.
		Contains internal flags to optionally use second-order exponential-Euler,
		though this is only implemented for solver order 0.
		'''

		def take_time_step(self, solver):
			verbose:bool = False
			use_exponential_euler:bool = False

			mesh = solver.mesh
			U = solver.state_coeffs.copy()
			U0 = solver.state_coeffs.copy()

			dt = solver.stepper.dt
			physics = solver.physics
			source_terms = physics.source_terms
			elem_helpers = solver.elem_helpers
			basis_val = elem_helpers.basis_val
			quad_wts = elem_helpers.quad_wts
			x_elems = elem_helpers.x_elems
			djac_elems = elem_helpers.djac_elems
			nq = quad_wts.shape[0]
			ns = physics.NUM_STATE_VARS
			nb = basis_val.shape[1]

			error_history = []

			if solver.order == 0 and use_exponential_euler:
				# Compute residual vector using pre-allocated space
				res = solver.get_residual(U, self.res)
				# Evaluate rate vector for state coefficients with shape (ne, nq, ns)
 				# mult_inv_mass_matrix(mesh, solver, self.dt, res)
				dU = dt * np.einsum('ijk, ikl -> ijl', solver.elem_helpers.iMM_elems, res)
				# Map coefficients to quadrature values with shape (ne, nq, ns,)
				Uq = helpers.evaluate_state(U, basis_val, skip_interp=solver.basis.skip_interp)
				Sjac = np.zeros([mesh.num_elems, nq, ns, ns])
				Sjac = physics.eval_source_term_jacobians(Uq, x_elems,
						solver.time, Sjac)
				
				# Exponential Euler in comments here (for element order 0 only)
				_A = np.einsum(
					'elj, ij, ik, eimn, i, ei -> elkmn',
					solver.elem_helpers.iMM_elems,
					basis_val, # Symmetric basis values at quad points
					basis_val, # Symmetric basis values at quad points
					Sjac,      # Source jacobian
					quad_wts.squeeze(axis=-1),
					djac_elems.squeeze(axis=-1))
				_b = np.swapaxes(dU/dt, 1, 2)[:,np.newaxis,np.newaxis,...]
				# [A b]
				# [0 0]
				enhanced_matrix = np.concatenate((
					np.concatenate((_A, _b,), axis=4),
					np.zeros((*_A.shape[0:3], 1, _A.shape[-1]+1)),), axis=3)
				# Compute phi_1(hJ) @ f
				delta_U_exp = np.zeros_like(U)
				for i in range(delta_U_exp.shape[0]):
					delta_U_exp[i,0,:] = scipy.linalg.expm(dt * enhanced_matrix[i,0,0,:,:])[...,0:ns,-1]
				U = U + dt * delta_U_exp
			else: # Not exponential Euler
				for i in range(1):
					# Compute residual vector using pre-allocated space
					res = solver.get_residual(U, self.res)
					# Evaluate rate vector for state coefficients with shape (ne, nq, ns)
					# mult_inv_mass_matrix(mesh, solver, self.dt, res)
					dU = dt * np.einsum('ijk, ikl -> ijl', solver.elem_helpers.iMM_elems, res)
					# Map coefficients to quadrature values with shape (ne, nq, ns,)
					Uq = helpers.evaluate_state(U, basis_val, skip_interp=solver.basis.skip_interp)
					# Evaluate source term Jacobian at qudarature points with shape (ne, nq, ns, ns,)
					if False: # np.all([s.__class__.__name__ == "FrictionVolFracVariableMu" for s in source_terms]):
						# Known sparsity pattern
						# Define backward Euler evolution matrix with shape (ne, nq, nq, ns, ns)
						A = np.eye(ns)[np.newaxis,np.newaxis,:,:] \
								- dt * np.einsum(
										'elj, ij, ik, eimn, i, ei -> elkmn',
										solver.elem_helpers.iMM_elems,
										basis_val, # Symmetric basis values at quad points
										basis_val, # Symmetric basis values at quad points
										Sjac,      # Source jacobian
										quad_wts.squeeze(axis=-1),
										djac_elems.squeeze(axis=-1))
						# Invert along last two axes (ne, nq, nq, *ns*, *ns*)
						iA = np.linalg.inv(A)
					else:
						Sjac = np.zeros([mesh.num_elems, nq, ns, ns])
						Sjac = physics.eval_source_term_jacobians(Uq, x_elems,
								solver.time, Sjac)
						# Define backward Euler evolution matrix with shape (ne, nq, nq, ns, ns)
						A = np.eye(ns)[np.newaxis,np.newaxis,:,:] \
								- dt * np.einsum(
										'elj, ij, ik, eimn, i, ei -> elkmn',
										solver.elem_helpers.iMM_elems,
										basis_val, # Symmetric basis values at quad points
										basis_val, # Symmetric basis values at quad points
										Sjac,      # Source jacobian
										quad_wts.squeeze(axis=-1),
										djac_elems.squeeze(axis=-1))
						# Invert along last two axes (ne, nq, nq, *ns*, *ns*)
						iA = np.linalg.inv(A)

					# Newton iteration for residual vector
					newton_residual = (U - U0 - dU)
					U -= np.einsum("ijkmn, ikn -> ijm", iA, newton_residual)

					# Error metric
					error_state_weights = np.ones((ns))
					error_state_weights[solver.physics.get_state_index("Energy")] = 1e-3
					error_metric = np.linalg.norm(newton_residual * error_state_weights, axis=(1,2)).max(axis=0)
					error_history.append(error_metric)

					if error_metric < 1e-1:
						break
				else:
					if verbose:
						print("Implicit Euler failed to converge.")

			# res = np.einsum('ijkll, ijl -> ikl', A, U) + dU
			# U = np.einsum('ijkll, ijl -> ikl', iA, res)

			solver.apply_limiter(U)
			solver.state_coeffs = U

			return res # [ne, nb, ns]


	class BDF1_Original(SourceStepperBase):
		'''
		1st-order Backward Differencing (BDF1) method inherits attributes
		from SourceStepperBase. See SourceStepperBase for detailed comments of methods and
		attributes.

		Additional methods and attributes are commented below.
		'''

		# constant used to differentiate between BDF1 (BETA=1.0) and linearize
		# Trapezoidal scheme (BETA=0.5) 
		BETA = 1.0

		def take_time_step(self, solver):
			mesh = solver.mesh
			U = solver.state_coeffs

			res = self.res

			res = solver.get_residual(U, res)
			dU = mult_inv_mass_matrix(mesh, solver, self.dt, res)

			A, iA = self.get_jacobian_matrix(mesh, solver)

			res = np.einsum('ijkll, ijl -> ikl', A, U) + dU
			U = np.einsum('ijkll, ijl -> ikl', iA, res)

			solver.apply_limiter(U)
			solver.state_coeffs = U

			return res # [ne, nb, ns]

		def get_jacobian_matrix(self, mesh, solver):
			'''
			Calculates the Jacobian matrix of the source term and its inverse for all elements
			Inputs:
			-------
				mesh: mesh object
				solver: solver object (e.g., DG, ADERDG, etc...)
			Outputs:
			--------
				A: matrix returned for linear solve [ne, nb, nb, ns, ns]
				iA: inverse matrix returned for linear solve
					[ne, nb, nb, ns, ns]
			'''
			basis = solver.basis
			nb = basis.nb
			physics = solver.physics
			U = solver.state_coeffs
			ns = physics.NUM_STATE_VARS

			iMM_elems = solver.elem_helpers.iMM_elems

			A, iA = self.get_jacobian_matrix_elems(solver, iMM_elems, U)

			return A, iA # [nelem, nb, nb, ns]

		def get_jacobian_matrix_elems(self, solver, iMM_elems, Uc):
			'''
			Calculates the Jacobian matrix of the source term and its
			inverse for each element. Definition of 'Jacobian' matrix:
			A = I - BETA*dt*iMM^{-1}*dRdU

			Inputs:
			-------
				solver: solver object (e.g., DG, ADERDG, etc...)
				elem_ID: element ID
				iMM: inverse mass matrix [ne, nb, nb]
				Uc: state coefficients [ne, nb, ns]
			Outputs:
			--------
				A: matrix returned for linear solve [ne, nb, nb, ns, ns]
				iA: inverse matrix returned for linear solve
					[ne, nb, nb, ns, ns]
			'''
			mesh = solver.mesh
			nelem = mesh.num_elems
			beta = self.BETA
			dt = solver.stepper.dt
			physics = solver.physics
			source_terms = physics.source_terms

			elem_helpers = solver.elem_helpers
			basis_val = elem_helpers.basis_val
			quad_wts = elem_helpers.quad_wts
			x_elems = elem_helpers.x_elems
			djac_elems = elem_helpers.djac_elems
			nq = quad_wts.shape[0]
			ns = physics.NUM_STATE_VARS
			nb = basis_val.shape[1]

			Uq = helpers.evaluate_state(Uc, basis_val,
					skip_interp=solver.basis.skip_interp) # [ne, nq, ns])

			# Evaluate the source term Jacobian [ne, nq, ns, ns]
			Sjac = np.zeros([nelem, nq, ns, ns])
			Sjac = physics.eval_source_term_jacobians(Uq, x_elems,
					solver.time, Sjac)

			# Call solver helper to get dRdU (see solver/tools.py)
			dRdU = solver_tools.calculate_dRdU(elem_helpers, Sjac)
				# [ne, nb, nb, ns, ns]

			# Define the identity matrix
			I = np.expand_dims(np.expand_dims(np.eye(nb), axis=2), axis=3)

			A = I - beta*dt * \
					np.einsum('eij, ejklm -> eiklm', iMM_elems, dRdU)

			iA = np.zeros_like(A)
			for i in range(ns):
				for s in range(ns):
					iA[:, :, :, i, s] = np.linalg.inv(A[:, :, :, i, s])

			return A, iA # [ne, nb, nb, ns, ns]

	class Trapezoidal(SourceStepperBase):
		'''
		2nd-order Trapezoidal method. The resulting nonlinear system is solved
		using Scipy rootfinding, as opposed to linearization, due to improved
		convergence properties.
		'''
		def take_time_step(self, solver):
			mesh = solver.mesh
			U = solver.state_coeffs

			mesh = solver.mesh
			elem_helpers = solver.elem_helpers
			basis_val = elem_helpers.basis_val
			iMM_elems = elem_helpers.iMM_elems
			quad_pts = elem_helpers.quad_pts
			quad_wts = elem_helpers.quad_wts
			x_elems = elem_helpers.x_elems

			Uq = helpers.evaluate_state(U, basis_val,
					skip_interp=solver.basis.skip_interp) # [ne, nq, ns])

			# Solve the nonlinear system to get the new solution. This is done
			# for each element and each quadrature point, fully uncoupled. This
			# is actually only valid for orthogonal bases - for nodal bases this
			# could incur some error.
			for i in range(Uq.shape[0]):
				for j in range(Uq.shape[1]):
					sol = scipy.optimize.root(self.rhs_sources, Uq[i, j], args=(solver,
							x_elems[i, j], Uq[i, j]))
					Uq[i, j] = sol.x

			res = self.res

			# Project onto the basis state from the quadrature points
			solver_tools.L2_projection(mesh, iMM_elems, solver.basis,
					quad_pts, quad_wts, Uq, U)

			solver.apply_limiter(U)
			solver.state_coeffs = U

			return res # [ne, nb, ns]

		def rhs_sources(self, Uq_new, solver, x, Uq):
			Uq = Uq.reshape((1, 1, -1))
			Uq_new = Uq_new.reshape((1, 1, -1))
			x = x.reshape((1, 1, -1))
			dt = solver.stepper.dt

			# Point at n
			Sq = np.zeros_like(Uq)
			Sq = solver.physics.eval_source_terms(Uq, x, solver.time,
					Sq)
			# Point at n+1
			Sq_new = np.zeros_like(Uq_new)
			Sq_new = solver.physics.eval_source_terms(Uq_new, x, solver.time,
					Sq_new)
			# Return RHS of trapezoid rule
			return (Uq_new - Uq - .5*dt*(Sq_new + Sq))[0, 0] # [ns]


	class LSODA(SourceStepperBase):
		'''
		Scipy LSODA solver. This solver switches between a high-order
		Adams-Bashforth scheme and BDF scheme depending on the stiffness
		of the system. Works for very stiff problems.

		Additional methods and attributes are commented below.
		'''
		def take_time_step(self, solver):
			mesh = solver.mesh
			elem_helpers = solver.elem_helpers
			basis_val = elem_helpers.basis_val
			iMM_elems = elem_helpers.iMM_elems
			quad_pts = elem_helpers.quad_pts
			quad_wts = elem_helpers.quad_wts
			x_elems = elem_helpers.x_elems

			U = solver.state_coeffs

			Uq = helpers.evaluate_state(U, basis_val,
					skip_interp=solver.basis.skip_interp) # [ne, nq, ns])

			res = self.res

			Uq0, t0 = Uq.reshape(-1), solver.time
			dt = self.dt

			subiterations = []

			# Instantiate ode object
			r = ode(self.rhs_sources, jac=None)
			r.set_integrator('lsoda', nsteps=50000, atol=1e-14, rtol=1e-12)
			r.set_initial_value(Uq0, t0).set_f_params(x_elems, Uq, 
					solver, subiterations)

			value = r.integrate(r.t+dt).reshape([res.shape[0], res.shape[1],
				res.shape[2]])

			subiterations = np.unique(subiterations)

			# Print the number of subiterations for each iteration and
			# store the total number of ODE subiterations for the solver
			print("Subiterations:", len(subiterations))
			solver.count_evaluations += len(subiterations)

			# Project onto the basis state from the quadrature points
			solver_tools.L2_projection(mesh, iMM_elems, solver.basis,
					quad_pts, quad_wts, value, U)

			solver.apply_limiter(U)
			solver.state_coeffs = U

			return res # [ne, nb, ns]

		def rhs_sources(self, t, Uq0, x, Uq, solver, subiterations):
			'''
			Internal RHS call for the ODE solver

			Inputs:
			-------
				t: current time
				Uq0: solution state at the quadrature points
					[ne x nq x ns]
				x: coordinates of the quadrature points [ne, nq, ndim]

			Outputs:
			--------
				Sq: Source term evaluated at the quadrature points
					[ne, nq, ns]
			'''
			# Append to the subiteration counter
			subiterations.append(t)

			# Reconstruct shapes for source term evaluation on the
			# quadrature points.
			Uq = Uq0.reshape(Uq.shape)
			Sq = np.zeros_like(Uq)

			# Evaluate source term on quadrature points
			Sq = solver.physics.eval_source_terms(Uq, x, t, Sq)

			return Sq.reshape(-1) # ode function requires stacked array