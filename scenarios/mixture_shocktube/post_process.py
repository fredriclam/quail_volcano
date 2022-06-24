import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np
import scipy.io

import meshing.tools as mesh_tools
import numerics.helpers.helpers as helpers
import numerics.basis.tools as basis_tools

import scipy.interpolate

# Set fname
# fnameRadical = "pocket_atmos_flared" + "_"
# fnameRadical = "pocket_atmos_mushroom" + "_"
# fnameRadical = "pocket_atmos_shroomC_Sod_debug" + "_"

solver = readwritedatafiles.read_data_file("mixture_shocktube_conduit_final.pkl")
mesh = solver.mesh
physics = solver.physics

plot.prepare_plot(linewidth=0.5)
plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
		create_new_figure=True, include_mesh=False, regular_2D=False, 
		show_elem_IDs=False)
# Save figure
plot.show_plot()
plot.save_figure(file_name='Pressure', file_type='pdf', crop_level=2)

plot.plot_solution(mesh, physics, solver, "SoundSpeed", plot_numerical=True, 
		create_new_figure=True, include_mesh=False, regular_2D=False, 
		show_elem_IDs=False)
plot.show_plot()
plot.save_figure(file_name='SoundSpeed', file_type='pdf', crop_level=2)

quit()

name_2D_domain = "referenceM"
num_2D_domains = 2 #!
dataSize = 1200+1
verbose = False

outputfname = "redo_" + name_2D_domain + "_"
# outputfname = "void_"
fnameRadicals = {
	"conduit": f"{name_2D_domain}_conduit",
}
for i in range(num_2D_domains):
	fnameRadicals[i] = f"{name_2D_domain}{i+1}"
for key in fnameRadicals.keys():
	if not fnameRadicals[key].endswith("_"):
		fnameRadicals[key] += "_"

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

	domain_rho = []
	domain_rhou = []
	domain_rhov = []
	domain_u = []
	domain_v = []
	domain_T = []
	domain_s = []
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
		domain_rho.append(getns('Density'))
		domain_rhou.append(getns('XMomentum'))
		domain_rhov.append(getns('YMomentum'))
		domain_u.append(domain_rhou[i] / domain_rho[i])
		domain_v.append(domain_rhov[i] / domain_rho[i])
		domain_T.append(getns('Temperature'))
		domain_s.append(getns('Entropy'))
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
		
		state_names = ['Density', 'Energy', 'XMomentum',
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
		"domain_rho": domain_rho,
		"domain_E": domain_E,
		"domain_rhou": domain_rhou,
		"domain_rhov": domain_rhov,
		"domain_s": domain_s,
		"domain_u": domain_u,
		"domain_v": domain_v,
		"domain_T": domain_T,
		"domain_normgradrho": domain_normgradrho,
		"domain_vorticity": domain_vorticity,
		"samplePointsConduit": samplePointsConduit,
		"pConduit": getns_conduit("Pressure"),
		"rhoConduit": getns_conduit("Density"),
		"EConduit": getns_conduit("Energy"),
		"sConduit": getns_conduit("Entropy"),
		"uConduit": getns_conduit("XMomentum") / getns_conduit("Density"),
		"TConduit": getns_conduit("Temperature"),
		"integrated": domainIntegrals,
		"t": solver.time,
		"x_r1": x_r1,
		"x_r2": x_r2,
	})
	
	print(f"Converted to .mat at {outputfname}.")

def pre_compute(fname_radical):
	# Load initial solution
	solver = readwritedatafiles.read_data_file(f"{fname_radical}0.pkl")
	# Get quadrature data for static mesh
	quad_pts, quad_wts = solver.mesh.gbasis.get_quadrature_data(
		solver.basis.get_quadrature_order(solver.mesh, 2*np.amax([solver.order, 1]),
		physics=solver.physics))
	djacs = np.zeros((solver.mesh.num_elems, quad_wts.shape[0]))

	# Get points at nodes of each elements
	# temp_order_ = solver.order
	# solver.order = 0
	# sample_points_in_domain = plot.get_sample_points(
	# 	solver.mesh, solver, solver.physics, solver.basis, True)
	# solver.order = temp_order_

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

precomps = [pre_compute(fnameRadicals[j]) for j in range(num_2D_domains)]
for i in range(dataSize):
	pkl2mat(
		[f"{fnameRadicals[j]}{i}.pkl" for j in range(num_2D_domains)],
		f"{fnameRadicals['conduit']}{i}.pkl",
	  precomps,
		outputfname=(outputfname + f"{i}"),
		verbose=False)













# # Compute L2 error
# post.get_error(mesh, physics, solver, "Entropy", 
# 		normalize_by_volume=False)

# # Unpack
# mesh = solver.mesh
# physics = solver.physics

# plot.prepare_plot(linewidth=0.5)
# plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
# 		create_new_figure=True, include_mesh=True, regular_2D=False, 
# 		show_elem_IDs=False)
# # Save figure
# plot.save_figure(file_name='Pressure', file_type='pdf', crop_level=2)

print("End of postprocessing.")

if False:
	''' Plot '''
	### Pressure contour ###
	plot.prepare_plot(linewidth=0.5)
	plot.plot_solution(mesh, physics, solver, "Pressure", plot_numerical=True, 
			create_new_figure=True, include_mesh=True, regular_2D=False, 
			show_elem_IDs=False)
	# Save figure
	plot.save_figure(file_name='Pressure', file_type='pdf', crop_level=2)

	### Entropy contour ###
	# plot.plot_solution(mesh, physics, solver, "Entropy", plot_numerical=True, 
	# 		create_new_figure=True, include_mesh=True, regular_2D=False)
	# # Save figure
	# plot.save_figure(file_name='Entropy', file_type='pdf', crop_level=2)

	### Boundary info ###
	# Plot pressure in x-direction along wall
	# Boundary integral gives drag force in x-direction
	# post.get_boundary_info(solver, mesh, physics, "y1", "Pressure", 
	# 		dot_normal_with_vec=True, vec=[1.,0.], integrate=True, 
	# 		plot_vs_x=True, plot_vs_y=False, fmt="bo", ylabel="$F_x$")

	plot.show_plot()
