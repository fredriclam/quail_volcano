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
#       File : src/solver/tools.py
#
#       Contains additional methods (tools) for the DG solver class
#
# ------------------------------------------------------------------------ #
import numpy as np
import sys

import general
import numerics.basis.tools as basis_tools
import numerics.helpers.helpers as helpers
import solver.tools as solver_tools

import physics.multiphasevpT.atomics as atomics


def set_function_definitions(solver, params):
	'''
	This function sets the necessary functions for the given case 
	dependent upon setter flags in the input deck (primarily for 
	the diffusive flux definitions)

	Inputs:
	-------
		solver: solver object
		params: dict with solver parameters
	'''
	if solver.physics.diff_flux_fcn:
		solver.evaluate_gradient = helpers.evaluate_gradient
		solver.ref_to_phys_grad = helpers.ref_to_phys_grad
		solver.calculate_boundary_flux_integral_sum = \
			solver_tools.calculate_boundary_flux_integral_sum
	else:
		solver.evaluate_gradient = general.pass_function
		solver.ref_to_phys_grad = general.pass_function
		solver.calculate_boundary_flux_integral_sum = \
			general.zero_function


def calculate_volume_flux_integral(solver, elem_helpers, Fq):
	'''
	Calculates the volume flux integral for the DG scheme

	Inputs:
	-------
		solver: solver object
		elem_helpers: helpers defined in ElemHelpers
		Fq: flux array evaluated at the quadrature points [ne, nq, ns, ndims]

	Outputs:
	--------
		res_elem: calculated residual array
			[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
			# [ne, nq, nb, ndims]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate flux quadrature
	F_quad = np.einsum('ijkl, jm, ijm -> ijkl', Fq, quad_wts, djac_elems)
			# [ne, nq, ns, ndims]
	# Calculate residual
	res_elem = np.einsum('ijnl, ijkl -> ink', basis_phys_grad_elems, F_quad)
			# [ne, nb, ns]
	return res_elem # [ne, nb, ns]


def calculate_boundary_flux_integral(basis_val, quad_wts, Fq):
	'''
	Calculates the boundary flux integral for the DG scheme

	Inputs:
	-------
		basis_val: basis function for the interior element [nf, nq, nb]
		quad_wts: quadrature weights [nq, 1]
		Fq: flux array evaluated at the quadrature points [nf, nq, ns]

	Outputs:
	--------
		resB: residual contribution (from boundary face) [nf, nb, ns]
	'''
	# Calculate flux quadrature
	Fq_quad = np.einsum('ijk, jm -> ijk', Fq, quad_wts) # [nf, nq, ns]

	# Calculate residual
	resB = np.einsum('ijn, ijk -> ink', basis_val, Fq_quad) # [nf, nb, ns]

	return resB # [nf, nb, ns]


def calculate_boundary_flux_integral_sum(basis_ref_grad, quad_wts, Fq):
	'''
	Calculates the directional boundary flux integrals for diffusion fluxes

	Inputs:
	-------
		basis_ref_grad: evaluated gradient of the basis function in 
			reference space [nq, nb, ndims]
		quad_wts: quadrature weights [nq, 1]
		Fq: Direction diffusion flux contribution [nf, nq, ns, ndims]

	Outputs:
	--------
		resB: residual contribution (from boundary face) [nf, nb, ns]
	'''

	# Calculate flux quadrature
	Fq_quad = np.einsum('ijkl, jm -> ijkl', Fq, quad_wts) # [nf, nq, ns, ndims]

	# Calculate residual
	resB = np.einsum('ijnl, ijkl -> ink', basis_ref_grad, Fq_quad)

	return resB # [nf, nb, ns]


def calculate_source_term_integral(elem_helpers, Sq):
	'''
	Calculates the source term volume integral for the DG scheme

	Inputs:
	-------
		elem_helpers: helpers defined in ElemHelpers
		Sq: source term array evaluated at the quadrature points [ne, nq, ns]

	Outputs:
	--------
		res_elem: calculated residual array (for volume integral of all elements)
		[ne, nb, ns]
	'''
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_val = elem_helpers.basis_val # [nq, nb]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]

	# Calculate source term quadrature
	Sq_quad = np.einsum('ijk, jm, ijm -> ijk', Sq, quad_wts, djac_elems)
			# [ne, nq, ns]

	# Calculate residual
	res_elem = np.einsum('jn, ijk -> ink', basis_val, Sq_quad) # [ne, nb, ns]

	return res_elem # [ne, nb, ns]

def calculate_artificial_viscosity_integral(physics, elem_helpers, Uc, av_param, p):
	'''
	Calculates the artificial viscosity volume integral, given in:
		Hartmann, R. and Leicht, T, "Higher order and adaptive DG methods for
		compressible flows", p. 92, 2013.

	Inputs:
	-------
		physics: physics object
		elem_helpers: helpers defined in ElemHelpers
		Uc: state coefficients of each element
		av_param: artificial viscosity parameter
		p: solution basis order

	Outputs:
	--------
		res_elem: artificial viscosity residual array for all elements
		[ne, nb, ns]
	'''
	# Unpack
	quad_wts = elem_helpers.quad_wts # [nq, 1]
	basis_phys_grad_elems = elem_helpers.basis_phys_grad_elems
			# [ne, nq, nb, dim]
	basis_val = elem_helpers.basis_val # [nq, nb]
	djac_elems = elem_helpers.djac_elems # [ne, nq, 1]
	vol_elems = elem_helpers.vol_elems # [ne]
	ndims = basis_phys_grad_elems.shape[3]

	# Evaluate solution at quadrature points
	Uq = helpers.evaluate_state(Uc, basis_val)
	# Evaluate solution gradient at quadrature points
	grad_Uq = np.einsum('ijnl, ink -> ijkl', basis_phys_grad_elems, Uc)
	
	# For Euler equations, use pressure as the smoothness variable
	if physics.PHYSICS_TYPE == general.PhysicsType.Euler:
		pressure = physics.compute_additional_variable("Pressure", Uq,
			flag_non_physical=False)[:, :, 0]
		# Compute pressure gradient
		grad_p = physics.compute_pressure_gradient(Uq, grad_Uq)
		# Compute its magnitude
		norm_grad_p = np.linalg.norm(grad_p, axis = 2)
		# Calculate smoothness switch
		f = norm_grad_p / (pressure + 1e-12)
	elif physics.PHYSICS_TYPE == general.PhysicsType.MultiphasevpT:
		# Apply a hydrostatic-relative limiting
		# Compute using atomics for speed
		arhoVec = Uq[..., physics.get_mass_slice()]
		rho = arhoVec.sum(axis=2, keepdims=True)
		rhou = Uq[..., physics.get_momentum_slice()]
		u = rhou / rho

		T = atomics.temperature(arhoVec, Uq[..., physics.get_momentum_slice()],
			Uq[..., physics.get_state_slice("Energy")], physics)
		gas_volfrac = atomics.gas_volfrac(arhoVec, T, physics)
		pressure = atomics.pressure(arhoVec, T, gas_volfrac, physics)

		# Additional weighting with approximate residual with pressure dominated
		# flow.
		# Strong form residual is (df/dU) dU/dx evaluated at the quadrature points.
		# Approximating momentum flux helps with condensed-state flow (no exsolution)
		if physics.NDIMS == 2:
			# use_legacy_AV = False
			# if use_legacy_AV:
			# 	# Legacy AV (large AVparam needed)
			# 	pgrad = physics.compute_pressure_gradient(Uq, grad_Uq)
			# 	f = np.linalg.norm(pgrad, axis=2) / (pressure[:, :, 0] + 1e-12)

			# Compute derivative of pressure w.r.t. state
			psgrad = atomics.pressure_sgrad2D(arhoVec, pressure, T,
				u[...,0:1], u[...,1:2], physics)
			# Compute hydrostatic-relative pressure gradient in-place
			pgrad_rel = np.einsum('ijk, ijkl -> ijl', psgrad, grad_Uq)
			pgrad_rel[...,1:2] -= -9.8 * arhoVec.sum(axis=-1, keepdims=True)
			f = np.linalg.norm(pgrad_rel, axis=2) / (pressure[:,:,0] + 1e-12)
			# Compute velocity gradient wrt state (ne, nq, ns, nd)
			usgrad = np.zeros((*psgrad.shape, 2))
			usgrad[:,:,physics.get_mass_slice(),:] = -np.expand_dims(u, axis=2)
			usgrad[:,:,physics.get_state_slice("XMomentum"),0] = 1 # du/(d(rho*u))
			usgrad[:,:,physics.get_state_slice("YMomentum"),1] = 1 # dv/(d(rho*v))
			usgrad /= np.expand_dims(rho, axis=2)

			# Direct computation of Fjac * divF, where Fjac has both x, y components
			# Convective flux part du/dq * q * dq/dx
			divF = np.einsum("ijm, ijnd, ijnd -> ijm", Uq, usgrad, grad_Uq,
				optimize=['einsum_path', (1, 2), (0, 1)])
			# Convective flux part u * dq/dx
			divF += np.einsum("ebd, ebid -> ebi", u, grad_Uq)
			# dp/dq part in momentum equation
			divF[..., physics.get_momentum_slice()] += np.einsum(
				"ebi, ebid -> ebd", psgrad, grad_Uq)
			# d(pu)/dq, d(pv)/dq parts in energy equation with dummy axis (index x)
			divF[..., physics.get_state_slice("Energy")] += np.einsum(
				"ebi, ebix -> ebx",
				u[...,0:1]*psgrad + pressure*usgrad[...,0],
				grad_Uq[...,0:1])
			divF[..., physics.get_state_slice("Energy")] += np.einsum(
				"ebi, ebix -> ebx",
				u[...,1:2]*psgrad + pressure*usgrad[...,1],
				grad_Uq[...,1:2])

			# Compute source in strong residual [ne, nq, ns]
			Sq = np.zeros_like(Uq) # [ne, nq, ns]
			Sq = physics.eval_source_terms(Uq, elem_helpers.x_elems,
				lambda: NotImplementedError(
					"Time dependence not implemented in Artificial Viscosity mod in tools.py"),
				Sq)
			# Compute strong form residual at each quadrature point (ne, nq, ns)
			Rh = -divF + Sq
			f *= np.einsum("ijm, ijm -> ij", psgrad, np.abs(Rh)) / pressure[...,0]
		elif physics.NDIMS == 1:
			psgrad = atomics.pressure_sgrad(arhoVec, pressure, T, u, physics)
			# psgrad = physics.compute_pressure_sgradient(Uq)
			pgrad = np.einsum('ijk, ijkl -> ijl', psgrad, grad_Uq)

			pgradhydro = -9.8 * arhoVec.sum(axis=-1, keepdims=True)
			f = np.linalg.norm(pgrad - pgradhydro, axis=2) / (pressure[:,:,0] + 1e-12)

			# Compute u gradient wrt state
			usgrad = np.zeros_like(psgrad)
			usgrad[:,:,physics.get_mass_slice()] = -u
			usgrad[:,:,physics.get_momentum_slice()] = 1
			usgrad /= rho

			is_computing_jacobian = False
			if is_computing_jacobian:
				# Assemble full flux jacobian (ne, nq, ns, ns) with convective flux part q*u
				Fjac = np.einsum("ijm, ijn -> ijmn", Uq, usgrad) \
					+ np.einsum("ebd, ij -> ebij", u, np.eye(physics.NUM_STATE_VARS))
				# Add dp/dq to row for (rho u)
				Fjac[:,:,physics.get_momentum_slice(),:] += np.expand_dims(psgrad,axis=2)
				# Add d(pu)/dq to row for e
				Fjac[:,:,physics.get_state_slice("Energy"),:] += \
					np.expand_dims(u*psgrad + pressure*usgrad,axis=2)
				# Compute spatial gradient of flux
				divF = np.einsum("ijmn, ijnd -> ijm", Fjac, grad_Uq)
			else:
				# Direct computation of Fjac * dq/dx
				# Convective flux part du/dq * q * dq/dx
				divF = np.einsum("ijm, ijn, ijnd -> ijm", Uq, usgrad, grad_Uq,
					optimize=['einsum_path', (1, 2), (0, 1)])
				# Convective flux part u * dq/dx
				divF += np.einsum("ebd, ebid -> ebi", u, grad_Uq)
				# dp/dq part in momentum equation
				divF[..., physics.get_momentum_slice()] += np.einsum(
					"ebi, ebid -> ebd", psgrad, grad_Uq)
				# d(pu)/dq part in energy equation
				divF[..., physics.get_state_slice("Energy")] += np.einsum(
					"ebi, ebix -> ebx", u*psgrad + pressure*usgrad, grad_Uq)

			# Compute source in strong residual [ne, nq, ns]
			Sq = np.zeros_like(Uq) # [ne, nq, ns]
			Sq = physics.eval_source_terms(Uq, elem_helpers.x_elems,
				lambda: NotImplementedError(
					"Time dependence not implemented in Artificial Viscosity mod in tools.py"),
				Sq)
			# Compute strong form residual at each quadrature point (ne, nq, ns)
			Rh = -divF + Sq

			# approx_strongform_res = (
			# 	pgrad
			# 	- rhou**2 / rho**2 * grad_Uq[..., 0:3].sum(axis=2, keepdims=True)[...,0] # - (rhou)^2 / rho^2 * d rho / dx
			# 	+ 2*rhou / rho * grad_Uq[:, :, 3:4, 0] # 2*rhou / rho * d (rho u )/dx
			# )
			# Approximate using only momentum term
			# Rp = approx_strongform_res[...,0] * physics.compute_pressure_sgradient(Uq)[...,3] / pressure
			f *= np.einsum("ijm, ijm -> ij", psgrad, np.abs(Rh)) / pressure[...,0]
		
		# f =  np.linalg.norm(grad_Uq[:, :, 0], axis=2) / (Uq[:, :, 0] + 1e-12) \
		# 	 + np.linalg.norm(grad_Uq[:, :, 1], axis=2) / (Uq[:, :, 1] + 1e-12) \
		# 	 + np.linalg.norm(grad_Uq[:, :, 2], axis=2) / (Uq[:, :, 2] + 1e-12) \
		# 	 + np.linalg.norm(grad_Uq[:, :, 3], axis=2) / (Uq[:, :, 3] + 1e-12) \
		# 	 + np.linalg.norm(physics.compute_pressure_gradient(Uq, grad_Uq), axis=2) / (pressure + 1e-12)

	# For everything else, use the first solution variable
	else:
		U0 = Uq[:, :, 0]
		grad_U0 = grad_Uq[:, :, 0]
		norm_grad_U0 = np.linalg.norm(grad_U0, axis = 2)
		# Calculate smoothness switch
		f =  norm_grad_U0 / (U0 + 1e-12)

	# Compute s_k
	s = np.zeros((Uc.shape[0], ndims))
	# Loop over dimensions
	for k in range(ndims):
		# Loop over number of faces per element
		for i in range(elem_helpers.normals_elems.shape[1]):
			# Integrate normals
			s[:, k] += np.einsum('jx, ij -> i', elem_helpers.face_quad_wts,
					np.abs(elem_helpers.normals_elems[:, i, :, k]))
		s[:, k] = 2 * vol_elems / s[:, k]
	# Compute h_k (the length scale in the kth direction)
	h = np.empty_like(s)
	# Loop over dimensions
	for k in range(ndims):
		h[:, k] = s[:, k] * (vol_elems / np.prod(s, axis=1))**(1/3) # TODO: Check 1/3 for ndims == 2. For ndims == 1, s==2V/2==V so pow (1/3) does nothing
	# Scale with polynomial order
	h_tilde = h / (p + 1)
	# Compute dissipation scaling
	epsilon = av_param *  np.einsum('ij, il -> ijl', f, h_tilde**3)
	# Calculate integral, with state coeffs factored out
	integral = np.einsum('ijm, ijpm, ijnm, jx, ijx -> ipn', epsilon,
				basis_phys_grad_elems, basis_phys_grad_elems, quad_wts,
				djac_elems)
	# Calculate residual
	res_elem = np.einsum('ipn, ipk -> ink', integral, Uc)

	return res_elem # [ne, nb, ns]


def calculate_dRdU(elem_helpers, Sjac):
	'''
	Helper function for ODE solvers that calculates the derivative of
	the source term integral with respect to the solution state.

	Inputs:
	-------
		elem_helpers: object containing precomputed element helpers
		Sjac: element source term Jacobian [ne, nq, ns, ns]

	Outputs:
	--------
		dRdU: derivative of the source term integral
			[ne, nb, nb, ns, ns]
	'''
	quad_wts = elem_helpers.quad_wts
	basis_val = elem_helpers.basis_val
	djac_elems = elem_helpers.djac_elems

	a = np.einsum('eijk, il, eil -> eijk', Sjac, quad_wts, djac_elems)

	return np.einsum('bq, ql, eqts -> eblts', basis_val.transpose(),
			basis_val, a)
		# [ne, nb, nb, ns, ns]


def mult_inv_mass_matrix(mesh, solver, dt, res):
	'''
	Multiplies the residual array with the inverse mass matrix

	Inputs:
		mesh: mesh object
		solver: solver object (e.g., DG, ADER-DG, etc...)
		dt: time step
		res: residual array

	Outputs:
		U: solution array
	'''
	physics = solver.physics
	iMM_elems = solver.elem_helpers.iMM_elems

	return dt*np.einsum('ijk, ikl -> ijl', iMM_elems, res)


def L2_projection(mesh, iMM, basis, quad_pts, quad_wts, f, U):
	'''
	Performs an L2 projection

	Inputs:
	-------
		mesh: mesh object
		iMM: space-time inverse mass matrix
		basis: basis object
		quad_pts: quadrature coordinates in reference space
		quad_wts: quadrature weights
		f: array of values to be projected from

	Outputs:
	--------
		U: array of values to be projected to
	'''
	if basis.basis_val.shape[0] != quad_wts.shape[0]:
		basis.get_basis_val_grads(quad_pts, get_val=True)

	for elem_ID in range(U.shape[0]):
		djac, _, _ = basis_tools.element_jacobian(mesh, elem_ID, quad_pts,
				get_djac=True)
		rhs = np.matmul(basis.basis_val.transpose(),
				f[elem_ID, :, :]*quad_wts*djac) # [nb, ns]

		U[elem_ID, :, :] = np.matmul(iMM[elem_ID], rhs)


def interpolate_to_nodes(f, U):
	'''
	Interpolates directly to the nodes of the element

	Inputs:
	-------
		f: array of values to be interpolated from

	Outputs:
	--------
		U: array of values to be interpolated onto
	'''
	U[:, :, :] = f


def get_ip_eta(mesh, order):
	i = order

	if i > 8:
		i = 8;
	etas = np.array([1., 4., 12., 12., 20., 30., 35., 45., 50.])

	return etas[i] * mesh.gbasis.NFACES


def update_progress(progress):
	'''
	Displays or updates a console progress bar.
	Accepts a float between 0 and 1. Any int will be converted to a float.
	A value under 0 represents a 'halt'.
	A value at 1 or bigger represents 100%.

	Inputs:
	-------
		progress: value representing the progress, scaled from 0 to 1
	'''
	# Length of the progress bar
	bar_length = 55

	status = ""
	# Convert ints
	if isinstance(progress, int):
		progress = float(progress)
	# Make sre it's a number
	if not isinstance(progress, float):
		progress = 0
		status = "error: progress var must be float\r\n"
	# Less than 0 'halts' the progress
	if progress < 0:
		progress = 0
		status = "Halt...\r\n"
	# Cap the progress at 100%
	if progress >= 1:
		progress = 1
		status = "Done...\r\n"

	# Compute number of blocks
	block = int(round(bar_length*progress))
	# Figure out the color
	if progress < .25:
		color = '\033[0;31m' # Dark red
	elif progress < .5:
		color = '\033[1;31m' # Light red
	elif progress < .75:
		color = '\033[0;33m' # Yellow
	elif progress < 1:
		color = '\033[0;32m' # Dark green
	else:
		color = '\033[1;32m' # Light green
	reset_color = '\033[0m'
	# Write out the text
	text = color + '\rPercent: [{0}] {1}% {2}'.format( "#"*block + "-"*(bar_length-block),
			int(round(progress*100)), status) + reset_color
	sys.stdout.write(text)
	sys.stdout.flush()
