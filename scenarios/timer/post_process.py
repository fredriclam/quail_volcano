import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np
import scipy.io

import meshing.tools as mesh_tools
import numerics.helpers.helpers as helpers
import numerics.basis.tools as basis_tools

# Set fname
# fnameRadical = "pocket_atmos_flared" + "_"
# fnameRadical = "pocket_atmos_mushroom" + "_"
# fnameRadical = "pocket_atmos_shroomC_Sod_debug" + "_"

name_2D_domain = "debug_standard3_generalB"
num_2D_domains = 6
dataSize = 250+1
verbose = False

outputfname = name_2D_domain + "_"
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
	samplePoints = []

	for i, input_fname in enumerate(input_fname_list):
		if verbose:
			print(f"Reading {input_fname}.")
		# Read in pickle file
		solver = readwritedatafiles.read_data_file(input_fname)
		# Unpack
		mesh = solver.mesh
		physics = solver.physics

		# Extract numerical solution on mesh
		samplePoints.append(plot.get_sample_points(
			mesh, solver, physics, solver.basis, True))
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
	for elem_ID in range(solver.mesh.num_elems):
			# Calculate element-local error
			djac, _, _ = basis_tools.element_jacobian(solver.mesh, elem_ID, 
				quad_pts, get_djac=True)
				
			djacs[elem_ID, :] = djac.squeeze()
	return (quad_pts, quad_wts, djacs)

precomps = [pre_compute(fnameRadicals[j]) for j in range(num_2D_domains)]
for i in range(dataSize):
	pkl2mat(
		[f"{fnameRadicals[j]}{i}.pkl" for j in range(num_2D_domains)],
		f"{fnameRadicals['conduit']}{i}.pkl",
	  precomps,
		outputfname=(outputfname + f"{i}"),
		verbose)













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
