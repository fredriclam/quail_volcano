import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np
import scipy.io

import meshing.tools as mesh_tools
import numerics.helpers.helpers as helpers
import numerics.basis.tools as basis_tools

import scipy.interpolate
import os

''' Set parameters '''
# Multidomain
use_multidomain = True
# Enter index of last iteration
dataSize = 490+1
# Conduit files to collate
fname_radical_conduit = "mixture_shocktube_conduit"
input_fname_conduit_list = [
	"mixture_shocktube_conduit",
	"mixture_shocktube_conduit2",
]
input_fname_2D_list = [
	"mixture_shocktube_atm1",
	# "mixture_shocktube_atm2",
]

outputfname = "v_mix0_"

''' Business logic '''

num_2D_domains = len(input_fname_2D_list)

def execute_post_process():
	''' Hoisted post process function. This is the only function called directly.'''
	precomps = [pre_compute(fname) for fname in input_fname_2D_list]
	for i in range(dataSize):
		# pkl2mat_both(
		# 	[f"{fname}_{i}.pkl" for fname in input_fname_2D_list],
		# 	[f"{fname}_{i}.pkl" for fname in input_fname_conduit_list],
		# 	precomps,
		# 	outputfname=(outputfname + f"{i}"),
		# 	verbose=False)
		pkl2mat_p_only([f"{fname}_{i}.pkl" for fname in input_fname_2D_list], (outputfname + f"{i}"))

def integrate(mesh, physics, solver, var_names, quad_pts, quad_wts, djacs):
	# Extract info
	U = solver.state_coeffs

	# Interpolate state to quadrature points
	solver.basis.get_basis_val_grads(quad_pts, True)
	u = helpers.evaluate_state(U, solver.basis.basis_val)

	outputs = np.zeros(len(var_names))
	for i in range(len(var_names)):
		# Computed requested quantity
		s = physics.compute_variable(var_names[i], u)

		# Loop through elements
		for elem_ID in range(mesh.num_elems):
			# Calculate element-local integral
			outputs[i] += np.sum(s[elem_ID]*quad_wts*np.expand_dims(djacs[elem_ID, :],axis=1))
	
	return outputs

def pkl2mat_both(input_fname_2D_list, input_fname_conduit_list, precompute_list, outputfname, verbose=False):

	Ne_per_domain = []
	xConduit = []
	pConduit = []
	prhoAConduit = []
	prhoWvConduit= []
	prhoMConduit = []
	rhoConduit = []
	EConduit = []
	prhoWtConduit = []
	prhoCConduit = []
	uConduit = []
	TConduit = []
	aConduit = []
	normgrad_prhoAConduit = []
	normgrad_pConduit = []
	phiConduit = []
	domain_prhoA = []
	domain_prhoWv = []
	domain_prhoM = []
	domain_rho = []
	domain_rhou = []
	domain_rhov = []
	domain_u = []
	domain_v = []
	domain_T = []
	domain_E = []
	domain_p = []
	domainIntegrals = []
	domain_normgradrho = []
	domain_vorticity = []
	samplePoints = []
	x_r1 = []
	x_r2 = []

	''' Get 2D state '''
	for i, input_fname in enumerate(input_fname_2D_list):
		if verbose:
			print(f"Reading {input_fname}.")
		# Read in pickle file
		solver = readwritedatafiles.read_data_file(input_fname)
		# Unpack
		mesh = solver.mesh
		physics = solver.physics
		# Extract numerical solution on mesh
		# Sample down to the corners
		temp_order_ = solver.order
		solver.order = 0
		samplePoints.append(plot.get_sample_points(
			mesh, solver, physics, solver.basis, True))
		solver.order = temp_order_

		# Get numerical solution
		getns = lambda quantity: plot.get_numerical_solution(
			physics,
			solver.state_coeffs,
			samplePoints[i],
			solver.basis,
			quantity)
		domain_prhoA.append(getns('pDensityA'))
		domain_prhoWv.append(getns('pDensityWv'))
		domain_prhoM.append(getns('pDensityM'))
		domain_rho.append(domain_prhoA[i] + domain_prhoWv[i] + domain_prhoM)
		domain_rhou.append(getns('XMomentum'))
		domain_rhov.append(getns('YMomentum'))
		domain_u.append(domain_rhou[i] / domain_rho[i])
		domain_v.append(domain_rhov[i] / domain_rho[i])
		domain_T.append(getns('Temperature'))
		domain_E.append(getns('Energy'))
		domain_p.append(getns('Pressure'))

		# Gradient quantities
		sample_basis_phys_grad_elems = precompute_list[i][3]
		domain_normgradrho.append(np.linalg.norm(
				np.einsum('ijnl, ink -> ijkl',
					np.array(sample_basis_phys_grad_elems),
					solver.state_coeffs)[:,:,0],
				axis=2, keepdims=True))
		du_ij = np.einsum('ijnl, ink -> ijkl',
					np.array(sample_basis_phys_grad_elems),
					solver.state_coeffs[:,:,1:3]/solver.state_coeffs[:,:,0:1])
		vorticity = du_ij[:,:,1:2,0] - du_ij[:,:,0:1,1]
		domain_vorticity.append(vorticity)

		# Integrate key quantities over entire domain
		quad_pts = precompute_list[i][0]
		quad_wts = precompute_list[i][1]
		djacs = precompute_list[i][2]
		
		state_names = ['pDensityA', 'pDensityWv', 'pDensityM',
			'Energy', 'XMomentum', 'YMomentum', 'Pressure', 'InternalEnergy']
		integrated_dict = dict(zip(
			state_names,
			integrate(solver.mesh, solver.physics, solver, 
								state_names, quad_pts, quad_wts, djacs)
		))
		domainIntegrals.append(integrated_dict)

		# Get special boundaries
		try:
			if i == 0:
				x_r1 = solver.bface_helpers.x_bgroups[
								solver.mesh.boundary_groups["r1"].number]
			elif i == 1:
				x_r2 = solver.bface_helpers.x_bgroups[
								solver.mesh.boundary_groups["r2"].number]
		except:
			print("2D names should be ordered r_1 < r_2. Failed to extract boundary location.")

	''' Get 1D state '''
	for fname in input_fname_conduit_list:
		# Get conduit state
		solver_conduit = readwritedatafiles.read_data_file(fname)
		conduitsolver = solver_conduit
		conduitmesh = solver_conduit.mesh
		conduitphysics = solver_conduit.physics
		samplePointsConduit = plot.get_sample_points(conduitmesh, 
			conduitsolver, conduitphysics, conduitsolver.basis, True)
		getns_conduit = lambda quantity: plot.get_numerical_solution(
			conduitphysics,
			conduitsolver.state_coeffs,
			samplePointsConduit,
			conduitsolver.basis,
			quantity)

		# Gradients
		normgrad_prhoA = np.linalg.norm(
				np.einsum('ijnl, ink -> ijkl',
					conduitsolver.elem_helpers.basis_phys_grad_elems,
					conduitsolver.state_coeffs)[:, :, 0],
				axis=2) # [ne, nq]
		# normgrad_prhoAConduit.append(np.expand_dims(scipy.interpolate.griddata(
		# 	( np.ndarray.flatten(conduitsolver.elem_helpers.x_elems[:,:,0]),
		# 		np.ndarray.flatten(conduitsolver.elem_helpers.x_elems[:,:,1]) 
		# 	),
		# 		np.ndarray.flatten(normgrad_prhoA),
		# 	samplePointsConduit,
		# 	method='cubic'
		# 	# method='nearest' # plot.get_sample_points(
		# 	# mesh, solver, physics, solver.basis, True), method='cubic')
		# ), 2))

		def interp_grad(quant):
			# Inteprolate gradient to quadrature points of state coefficients
			return scipy.interpolate.griddata(
				np.ndarray.flatten(conduitsolver.elem_helpers.x_elems[:,:,0]),
				np.ndarray.flatten(quant),
				samplePointsConduit,
				method='cubic'
			)
		
		normgrad_prhoA = np.linalg.norm(
				np.einsum('ijnl, ink -> ijkl',
					conduitsolver.elem_helpers.basis_phys_grad_elems,
					conduitsolver.state_coeffs)[:, :, 0],
				axis=2) # [ne, nq]
		normgrad_prhoAConduit.append(interp_grad(normgrad_prhoA))
		
		normgrad_p = np.linalg.norm(conduitphysics.compute_pressure_gradient(conduitsolver.state_coeffs, np.einsum('ijnl, ink -> ijkl',
					conduitsolver.elem_helpers.basis_phys_grad_elems,
					conduitsolver.state_coeffs)), axis=2)
		normgrad_pConduit.append(interp_grad(normgrad_p))

		xConduit.append(samplePointsConduit)
		pConduit.append(getns_conduit("Pressure"))
		prhoAConduit.append(getns_conduit("pDensityA"))
		prhoWvConduit.append(getns_conduit("pDensityWv"))
		prhoMConduit.append(getns_conduit("pDensityM"))
		rhoConduit.append(getns_conduit("pDensityA") + getns_conduit("pDensityWv")
										 + getns_conduit("pDensityM"))
		EConduit.append(getns_conduit("Energy"))
		uConduit.append(getns_conduit("XMomentum") / (getns_conduit("pDensityA") 
									+ getns_conduit("pDensityWv")
									+ getns_conduit("pDensityM")))
		prhoWtConduit.append(getns_conduit("pDensityWt"))
		prhoCConduit.append(getns_conduit("pDensityC"))
		TConduit.append(getns_conduit("Temperature"))
		aConduit.append(getns_conduit("SoundSpeed"))
		phiConduit.append(getns_conduit("phi"))
		Ne_per_domain.append(conduitmesh.num_elems)
		# t should be the same for each coupled domain
		t = conduitsolver.time

	# Get output file name if not specified
	postfix = f".mat"
	if outputfname is None:
		outputfname = "./" + input_fname_conduit_list[0][:-4] + ".mat"
	else:
		outputfname += postfix

	scipy.io.savemat(outputfname, mdict={
		"xConduit": xConduit,
		"Ne_per_domain": Ne_per_domain,
		"samplePointsConduit": samplePointsConduit,
		"pConduit": pConduit,
		"prhoAConduit": prhoAConduit,
		"prhoWvConduit": prhoWvConduit,
		"prhoMConduit": prhoMConduit,
		"rhoConduit": rhoConduit,
		"EConduit": EConduit,
		"uConduit": uConduit,
		"prhoWtConduit": prhoWtConduit,
		"prhoCConduit": prhoCConduit,
		"TConduit": TConduit,
		"aConduit": aConduit,
		"normgrad_prhoAConduit": normgrad_prhoAConduit,
		"normgrad_pConduit": normgrad_pConduit,
		"phiConduit": phiConduit,
		"t": t,
		"domain_prhoA": domain_prhoA,
		"domain_prhoWv": domain_prhoWv,
		"domain_prhoM": domain_prhoM,
		"domain_rho": domain_rho,
		"domain_rhou": domain_rhou,
		"domain_rhov": domain_rhov,
		"domain_u": domain_u,
		"domain_v": domain_v,
		"domain_T": domain_T,
		"domain_E": domain_E,
		"domain_p": domain_p,
		"x_r1": x_r1,
		"x_r2": x_r2,
		"samplePoints": samplePoints,
		"domain_normgradrho": domain_normgradrho,
		"domain_vorticity": domain_vorticity,
		"samplePointsConduit": samplePointsConduit,
		"integrated": domainIntegrals,
	})
	
	print(f"Converted to .mat at {outputfname} in folder {os.getcwd()}.")

def pkl2mat(input_fname_list, input_fname_conduit,
						precompute_list, outputfname=None, verbose=False):
	''' Converts loose .pkl files into .mat with all domains specified.

	Some additional variables:
		Pressure = "p"
		Temperature = "T"
		Entropy = "s"
		InternalEnergy = "\\rho e"
		TotalEnthalpy = "H"
		SoundSpeed = "c"
		MaxWaveSpeed = "\\lambda"
		Velocity = "|u|"
	'''

	domain_prhoA = []
	domain_prhoWv = []
	domain_prhoM = []
	domain_rho = []
	domain_rhou = []
	domain_rhov = []
	domain_u = []
	domain_v = []
	domain_T = []
	domain_E = []
	domain_p = []
	domainIntegrals = []
	domain_normgradrho = []
	domain_vorticity = []
	samplePoints = []
	x_r1 = []
	x_r2 = []

	for i, input_fname in enumerate(input_fname_list):
		if verbose:
			print(f"Reading {input_fname}.")
		# Read in pickle file
		solver = readwritedatafiles.read_data_file(input_fname)
		# Unpack
		mesh = solver.mesh
		physics = solver.physics

		# Extract numerical solution on mesh
		# Sample down to the corners
		temp_order_ = solver.order
		solver.order = 0
		samplePoints.append(plot.get_sample_points(
			mesh, solver, physics, solver.basis, True))
		solver.order = temp_order_

		# Get numerical solution
		getns = lambda quantity: plot.get_numerical_solution(
			physics,
			solver.state_coeffs,
			samplePoints[i],
			solver.basis,
			quantity)
		domain_prhoA.append(getns('pDensityA'))
		domain_prhoWv.append(getns('pDensityWv'))
		domain_prhoM.append(getns('pDensityM'))
		domain_rho.append(domain_prhoA[i] + domain_prhoWv[i] + domain_prhoM)
		domain_rhou.append(getns('XMomentum'))
		domain_rhov.append(getns('YMomentum'))
		domain_u.append(domain_rhou[i] / domain_rho[i])
		domain_v.append(domain_rhov[i] / domain_rho[i])
		domain_T.append(getns('Temperature'))
		domain_E.append(getns('Energy'))
		domain_p.append(getns('Pressure'))

		# More computed quantities
		if False:
			normgradrho = np.linalg.norm(
				np.einsum('ijnl, ink -> ijkl',
					solver.elem_helpers.basis_phys_grad_elems,
					solver.state_coeffs)[:, :, 0],
				axis=2) # [ne, nq]
			# print("Sample points: ")
			# print(plot.get_sample_points(
					# mesh, solver, physics, solver.basis, True).shape)
			# print("|rho|: ")
			# print(normgradrho.shape)
			# print("Uq")
			# print(solver.state_coeffs.shape)
			# domain_normgradrho.append(plot.interpolate_2D_soln_to_points(
			# 	physics,
			# 	plot.get_sample_points(
			# 		mesh, solver, physics, solver.basis, True),
			# 	normgradrho,
			# 	samplePoints[i]))
			domain_normgradrho.append(np.expand_dims(scipy.interpolate.griddata(
				( np.ndarray.flatten(solver.elem_helpers.x_elems[:,:,0]),
					np.ndarray.flatten(solver.elem_helpers.x_elems[:,:,1]) ),
				np.ndarray.flatten(normgradrho),
				samplePoints[i],
				method='nearest' # plot.get_sample_points(
				# mesh, solver, physics, solver.basis, True), method='cubic')
			), 2))

			gradu = np.einsum('ijnl, ink -> ijkl',
					solver.elem_helpers.basis_phys_grad_elems,
					solver.state_coeffs[:,:,1:3] / solver.state_coeffs[:,:,0:1])
			vorticity = gradu[:,:,1,0] - gradu[:,:,0,1]
			# domain_vorticity.append(plot.interpolate_2D_soln_to_points(
			# 	physics,
			# 	plot.get_sample_points(
			# 		mesh, solver, physics, solver.basis, True),
			# 	vorticity,
			# 	samplePoints[i]))
			domain_vorticity.append(np.expand_dims(scipy.interpolate.griddata(
				( np.ndarray.flatten(solver.elem_helpers.x_elems[:,:,0]),
					np.ndarray.flatten(solver.elem_helpers.x_elems[:,:,1]) ),
				np.ndarray.flatten(vorticity),
				samplePoints[i],
				method='nearest'  #plot.get_sample_points(
				# mesh, solver, physics, solver.basis, True), method='cubic'))
			), 2))

		# Gradient quantities
		sample_basis_phys_grad_elems = precompute_list[i][3]
		domain_normgradrho.append(np.linalg.norm(
				np.einsum('ijnl, ink -> ijkl',
					np.array(sample_basis_phys_grad_elems),
					solver.state_coeffs)[:,:,0],
				axis=2, keepdims=True))
		du_ij = np.einsum('ijnl, ink -> ijkl',
					np.array(sample_basis_phys_grad_elems),
					solver.state_coeffs[:,:,1:3]/solver.state_coeffs[:,:,0:1])
		vorticity = du_ij[:,:,1:2,0] - du_ij[:,:,0:1,1]
		domain_vorticity.append(vorticity)

		# Integrate key quantities over entire domain
		quad_pts = precompute_list[i][0]
		quad_wts = precompute_list[i][1]
		djacs = precompute_list[i][2]
		
		state_names = ['pDensityA', 'pDensityWv', 'pDensityM', 'Energy', 'XMomentum',
				'YMomentum', 'Pressure', 'InternalEnergy']
		integrated_dict = dict(zip(
			state_names,
			integrate(solver.mesh, solver.physics, solver, 
								state_names, quad_pts, quad_wts, djacs)
		))
		
		domainIntegrals.append(integrated_dict)

		# Get special boundaries
		if i == 0:
			x_r1 = solver.bface_helpers.x_bgroups[
							 solver.mesh.boundary_groups["r1"].number]
		elif i == 1:
			x_r2 = solver.bface_helpers.x_bgroups[
							 solver.mesh.boundary_groups["r2"].number]


	# Get conduit state
	solver_conduit = readwritedatafiles.read_data_file(input_fname_conduit)
	conduitsolver = solver_conduit
	conduitmesh = solver_conduit.mesh
	conduitphysics = solver_conduit.physics
	samplePointsConduit = plot.get_sample_points(conduitmesh, 
		conduitsolver, conduitphysics, conduitsolver.basis, True)
	getns_conduit = lambda quantity: plot.get_numerical_solution(
		conduitphysics,
		conduitsolver.state_coeffs,
		samplePointsConduit,
		conduitsolver.basis,
		quantity)

	# print(getns_conduit("Pressure")[-1,:])

	# Get output file name
	postfix = f".mat"
	if outputfname is None:
		outputfname = "./" + input_fname[:-4] + ".mat"
	else:
		outputfname += postfix
	

	scipy.io.savemat(outputfname, mdict={
		"samplePoints": samplePoints,
		"domain_p": domain_p,
		"domain_prhoA": domain_prhoA,
		"domain_prhoWv": domain_prhoWv,
		"domain_prhoM": domain_prhoM,
		"domain_rho": domain_rho,
		"domain_E": domain_E,
		"domain_rhou": domain_rhou,
		"domain_rhov": domain_rhov,
		"domain_u": domain_u,
		"domain_v": domain_v,
		"domain_T": domain_T,
		"domain_normgradrho": domain_normgradrho,
		"domain_vorticity": domain_vorticity,
		"samplePointsConduit": samplePointsConduit,
		"pConduit": getns_conduit("Pressure"),
		"prhoAConduit": getns_conduit("pDensityA"),
		"prhoWvConduit": getns_conduit("pDensityWv"),
		"prhoMConduit": getns_conduit("pDensityM"),
		"rhoConduit": getns_conduit("pDensityA") + getns_conduit("pDensityWv") \
									+ getns_conduit("pDensityM"),
		"EConduit": getns_conduit("Energy"),
		"uConduit": getns_conduit("XMomentum") / (getns_conduit("pDensityA") 
									+ getns_conduit("pDensityWv")
									+ getns_conduit("pDensityM")),
		"TConduit": getns_conduit("Temperature"),
		"integrated": domainIntegrals,
		"t": conduitsolver.time,
		"x_r1": x_r1,
		"x_r2": x_r2,
	})
	
	print(f"Converted to .mat at {outputfname}.")

def pre_compute(fname_radical):
	# Load initial solution
	solver = readwritedatafiles.read_data_file(f"{fname_radical}_0.pkl")
	# Get quadrature data for static mesh
	quad_pts, quad_wts = solver.mesh.gbasis.get_quadrature_data(
		solver.basis.get_quadrature_order(solver.mesh, 2*np.amax([solver.order, 1]),
		physics=solver.physics))
	djacs = np.zeros((solver.mesh.num_elems, quad_wts.shape[0]))

	sample_basis_phys_grad_elems = []
	for elem_ID in range(solver.mesh.num_elems):
			# Calculate element-local error
			djac, _, _ = basis_tools.element_jacobian(solver.mesh, elem_ID, 
				quad_pts, get_djac=True)
				
			djacs[elem_ID, :] = djac.squeeze()

			sample_points_ref = solver.basis.PRINCIPAL_NODE_COORDS
			sample_points_phys = mesh_tools.ref_to_phys(solver.mesh,
					elem_ID,
					solver.basis.PRINCIPAL_NODE_COORDS)
			nq_sample = sample_points_ref.shape[0]
			
			# Compute gradient of basis
			solver.basis.get_basis_val_grads(
				sample_points_ref, # sample_points_in_domain[elem_ID,:,:],
				get_val=True,
				get_ref_grad=True,
				get_phys_grad=True,
				ijac=solver.elem_helpers.ijac_elems[elem_ID,0:nq_sample,:,:])
			# Append gradient, in physical space, of the bases for element elem_ID
			sample_basis_phys_grad_elems.append(solver.basis.basis_phys_grad) # [nq=3,nb,dim]

	return (quad_pts, quad_wts, djacs, sample_basis_phys_grad_elems)

def pkl2mat_p_only(input_fname_2D_list, outputfname):
	samplePoints = []
	domain_p = []
	for i, input_fname in enumerate(input_fname_2D_list):
		solver = readwritedatafiles.read_data_file(input_fname)
		# Unpack
		mesh = solver.mesh
		physics = solver.physics
		# Extract numerical solution on mesh
		# Sample down to the corners
		temp_order_ = solver.order
		solver.order = 0
		samplePoints.append(plot.get_sample_points(
			mesh, solver, physics, solver.basis, True))
		solver.order = temp_order_

		# Get numerical solution
		getns = lambda quantity: plot.get_numerical_solution(
			physics,
			solver.state_coeffs,
			samplePoints[i],
			solver.basis,
			quantity)
		domain_p.append(getns('Pressure'))
		# Get output file name if not specified
		postfix = f".mat"
	if outputfname is None:
		outputfname = "./" + input_fname_conduit_list[0][:-4] + ".mat"
	else:
		outputfname += postfix

	scipy.io.savemat(outputfname, mdict={
		"domain_p": domain_p,
		"samplePoints": samplePoints,
	})

execute_post_process()