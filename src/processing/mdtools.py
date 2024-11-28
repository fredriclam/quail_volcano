	# Multidomain postprocessing toolkit
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import numpy as np
import processing.readwritedatafiles as readwritedatafiles
import processing.plot as plot
import traceback

import glob
from pickle import UnpicklingError
import matplotlib.tri as tri
from numpy.typing import ArrayLike

def viz(solver, plot_qty:str="Pressure", levels=None, clims=None):
	x, qty = downsample(solver, plot_qty)
	if solver.physics.NDIMS == 2:
		custom_plot_2D(x, qty, solver, levels=levels, clims=clims)
	elif solver.physics.NDIMS == 1:
		plt.plot(x.ravel(), qty.ravel(), '.-')

def downsample(solver, plot_qty:str="Pressure"):
	solver = copy.deepcopy(solver)
	# Downsample
	original_order = solver.order
	solver.order = 0
	equidistant_pts = True
	x = plot.get_sample_points(solver.mesh, solver, solver.physics, solver.basis,
				equidistant_pts)
	solver.order = original_order
	# Compute on downsampled points
	var_plot = plot.get_numerical_solution(solver.physics, solver.state_coeffs, x,
					solver.basis, plot_qty)
	return x, var_plot

def custom_plot_2D(x, var_plot, solver, levels=None, clims=None):
	# plot.plot_2D_general(solver.physics, x, var_plot)

	if levels is None:
		''' Open temp figure to get default levels'''
		figtmp = plt.figure()
		# Run this for the sole purpose of getting default contour levels
		levels = plot.plot_2D_regular(solver.physics, np.copy(x), np.copy(var_plot))
		if clims is not None:
			plt.clim(clims)
			levels = plot.plot_2D_regular(solver.physics, np.copy(x), np.copy(var_plot))
		plt.close(figtmp)

	num_elems = x.shape[0]
	for elem_ID in range(num_elems):
		# Triangulate each element one-by-one
		tris, utri = plot.triangulate(solver.physics, x[elem_ID], var_plot[elem_ID])
		# Plot
		plt.tricontourf(tris, utri, levels=levels, extend="both")
		if np.mod(elem_ID,100) == 0:
			plt.pause(0.05)

	# Adjust plot
	plt.gca().set_aspect('equal', adjustable='box')
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	return var_plot

def plot_mean(x, q, clims):
	''' Create very cheap plot '''
	cmap = plt.get_cmap()
	cnorm = matplotlib.colors.Normalize(vmin=clims[0], vmax=clims[1]) 
	for i in range(x.shape[0]):
		pp = plt.Polygon([[x[i,0,0], x[i,0,1]],
			[x[i,1,0], x[i,1,1]],
			[x[i,2,0], x[i,2,1]]], facecolor=cmap(cnorm(q[i,:,0].mean())), linewidth=0)
		plt.gca().add_patch(pp)
	plt.axis("auto")
	plt.axis("equal")

def compute_ref_mapping(x_tri):
	''' Compute jacobian
	x_tri: triangle nodes (..., 3, 2)

	Returns
	Matrix collection (..., 2, 2) such that reference coordinates can be
	obtained from

		M @ (x - x_tri[0,:]).

	'''
	_a00 = x_tri[...,1:2,0:1] - x_tri[...,0:1,0:1]
	_a01 = x_tri[...,2:3,0:1] - x_tri[...,0:1,0:1]
	_a10 = x_tri[...,1:2,1:2] - x_tri[...,0:1,1:2]
	_a11 = x_tri[...,2:3,1:2] - x_tri[...,0:1,1:2]
	_M = np.zeros((*x_tri.shape[:-2], 2, 2))
	_M[...,0:1,0:1] = _a11
	_M[...,0:1,1:2] = -_a01
	_M[...,1:2,0:1] = -_a10
	_M[...,1:2,1:2] = _a00
	_M /= (_a00 * _a11 - _a01 * _a10)
	return _M

def eval_state_at(U_loc:np.array, ref_coords_loc:np.array, order:int):
	'''
	U_loc: state coefficients for element (..., nb, ns)
	ref_coords_loc: 2D reference coordinates local to element (..., 2); shape has
	  to be broadcastable when multiplied against U_loc
	order: polynomial order in element
	'''

	# Check size of U_loc compared to expected size for a given order
	if order > 2 or order < 0:
		raise NotImplementedError("Only implemented for orders 0 to 2.")
	else:
		expected_num_dof = [1, 3, 6]
		if U_loc.shape[-2] != expected_num_dof[order]:
			raise ValueError(f"U_loc has axis -2 of size {U_loc.shape[-2]}, but "
				+ f" expected size {expected_num_dof[order]} for order {order}.") 
	# Pad axis for state-index
	ref_coords_loc = ref_coords_loc[...,np.newaxis]
	# Evaluate state by summing coefficients against basis functions
	_x = ref_coords_loc[...,0,:]
	_y = ref_coords_loc[...,1,:]
	if order == 0:
		U_sample = U_loc[...,0,:] * np.ones_like(_x) # Prepand new axis to match other order shapes
	elif order == 1:
		U_sample = (U_loc[...,0,:] * (1 - _x - _y)
		+ U_loc[...,1,:] * _x
		+ U_loc[...,2,:] * _y)
	elif order == 2:
		U_sample = (U_loc[...,0,:] * (1 - 2*_x - 2*_y) * (1 - _x - _y)
		+ U_loc[...,1,:] * 4 * _x * (1 - _x - _y)
		+ U_loc[...,2,:] * _x * (2*_x - 1)
		+ U_loc[...,3,:] * 4 * _y * (1 - _x - _y)
		+ U_loc[...,4,:] * 4 * _x * _y
		+ U_loc[...,5,:] * _y * (2*_y - 1)
		)
	else:
		raise NotImplementedError("Only implemented for orders 0 to 2.")
	return U_sample

def plot_detailed(x_tri, U, quantity_fn:callable, clims:tuple, order:int, cmap=None, solver=None,
									xview:tuple=None, yview:tuple=None,
									colorbar_label:str="value", n_samples:int=3, ax=None,
									add_colorbar=True, coordinate_scaling:callable=None,
									postproc_filter:callable=None):
	''' Create plot with higher sampling in element.
	For order == 1, n_samples only needs to be 2 (bilinear interpolation is used
	while drawing the gradient).

	x_tri: Triangle points: (ne,3,2)
	U:     State coefficients at nodes (ne, nb, ns), ordered in the reference
	       triangle in dictionary order for (y_ref, x_ref)
	quantity_fn: Quantity function evaluated at the sampling points (e.g. U -> p)
		 '''

	if solver is not None:
		# Retrieve x nodes from solver
		x_tri = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs]
	# Retrieve cmap if not provided
	cmap = plt.get_cmap() if cmap is None else cmap

	# Restrict draw to elements partially in viewport
	if xview is None:
		xview = (x_tri[...,0].min(), x_tri[...,0].max())
	if yview is None:
		yview = (x_tri[...,1].min(), x_tri[...,1].max())
	
	if ax is None:
		ax = plt.gca()

	# Restrict rendering to points at least partially in the viewport
	x_in_viewport = (x_tri[:,:,0:1] >= xview[0]) & (x_tri[:,:,0:1] <= xview[1])
	y_in_viewport = (x_tri[:,:,1:2] >= yview[0]) & (x_tri[:,:,1:2] <= yview[1])
	render_indices = np.where(np.any(x_in_viewport & y_in_viewport, axis=1))[0]

	# Compute reference mapping matrix
	ref_mapping = compute_ref_mapping(x_tri)

	for i in render_indices:
		# Establish bounding box
		xmin, xmax = x_tri[i,:,0].min(), x_tri[i,:,0].max()
		ymin, ymax = x_tri[i,:,1].min(), x_tri[i,:,1].max()
		# Sample grid points xbox in image order (x as normal, y downward)
		xbox = np.stack(np.meshgrid(
			np.linspace(xmin, xmax, n_samples),
			np.linspace(ymax, ymin, n_samples), indexing="xy"),
			axis=-1)
		# Compute reference coordinates for this element (ny=n_samples, nx=n_samples, 2)
		ref_coords_loc = np.einsum("ij, ...j -> ...i",
														   ref_mapping[i,...], xbox - x_tri[i,0,:])
		f = eval_state_at(U[i,:,:], ref_coords_loc, order)
		# Compute state -> quantity at sample points
		# f has shape (ny=n_samples, nx=n_samples, ns)
		f = quantity_fn(f)
		# Apply optional coordinate scaling
		if coordinate_scaling is not None:
			f *= coordinate_scaling(xbox)
		# Apply optional post-processing filter
		if postproc_filter is not None:
			f = postproc_filter(f)
		# Render triangle
		img = ax.imshow(f,
										extent=[xmin, xmax, ymin, ymax],
										vmin=clims[0],
										vmax=clims[1],
										interpolation='bilinear',
										cmap=cmap)
		polygon = Polygon(x_tri[i, :, :], closed=True,
											facecolor='none',
											edgecolor='none')
		ax.add_patch(polygon)
		img.set_clip_path(polygon)

	plt.axis("auto")
	plt.axis("equal")

	cb = None
	if add_colorbar:
		sm = plt.cm.ScalarMappable(
			norm=matplotlib.colors.Normalize(vmin=clims[0], vmax=clims[1]),
			cmap=cmap)
		cb = plt.colorbar(sm)
		cb.set_label(colorbar_label)

	return (ax, cb)

def plot_mean1D(x, q, clims, xscale=1.0, xshift=0.0):
	''' Create very cheap plot '''
	cmap = plt.get_cmap()
	cnorm = matplotlib.colors.Normalize(vmin=clims[0], vmax=clims[1]) 
	for i in range(x.shape[0]):
		pp = plt.Rectangle((xshift, x[i,0,0]), width=xscale, height=x[i,1,0]-x[i,0,0],
			facecolor=cmap(cnorm(q[i,:,0].mean())), linewidth=0)
		plt.gca().add_patch(pp)
	plt.axis("auto")
	plt.axis("equal")

def generate_anim(atm_names, conduit_names, outfilename, num_frames,
	plot_qty, filter=lambda x, x0:x, initial=1, stride=1, clims=None,
	is_high_detail=False):
	# This line needs to be here, with path to your ffmpeg path
	# plt.rcParams['animation.ffmpeg_path'] = "C:\\Users\\Fredric\\Documents\\ffmpeg\\ffmpeg-n4.4-latest-win64-gpl-4.4\\bin"
	if is_high_detail:
		print("High detail plot not implemented lol (defaulting to element-mean plotting)")
	fig = plt.figure(figsize=(11, 11), dpi=80)
	

	FFwriter = animation.FFMpegWriter()
	FFsetup_failure = False
	try:
		FFwriter.setup(fig, f"{outfilename}.mp4")
	except:
		print("Failed to set up FFWriter. Showing plots without saving animation.")
		traceback.print_exc()
		FFsetup_failure = True

	atm_initials = [readwritedatafiles.read_data_file(f"{name}_{0}.pkl") 
		for name in atm_names]
	conduit_initials = [readwritedatafiles.read_data_file(f"{name}_{0}.pkl") 
		for name in conduit_names]

	''' Compute values for each frame '''
	all_values = []
	all_x = []
	all_t = np.zeros((num_frames,))
	for i in range(num_frames):
		read_index = initial+i*stride
		all_values.append([])

		for dom_idx, name in enumerate(atm_names):
			solver = readwritedatafiles.read_data_file(f"{name}_{read_index}.pkl")
			# Compute quantity
			if plot_qty == "MassFlux":
				''' Mass flux magnitude '''
				x, px = downsample(solver, "XMomentum")
				x, py = downsample(solver, "YMomentum")
				qty = np.sqrt(px**2+py**2)
			else:
				x, qty = downsample(solver, plot_qty)
			# Compute quantity in initial condition
			_, qty_init = downsample(atm_initials[dom_idx], plot_qty)
			# Apply filter f(q, q0)
			qty = filter(qty, qty_init)
			all_values[i].append(qty)
			if i == 0:
				all_x.append(x)
			all_t[i] = solver.time


		for dom_idx, name in enumerate(conduit_names):
			solver = readwritedatafiles.read_data_file(f"{name}_{read_index}.pkl")
			# Compute quantity
			if plot_qty == "MassFlux":
				''' Mass flux magnitude '''
				x, px = downsample(solver, "XMomentum")
				qty = np.sqrt(px**2)
			else:
				x, qty = downsample(solver, plot_qty)
			# Compute quantity in initial condition
			_, qty_init = downsample(conduit_initials[dom_idx], plot_qty)
			# Apply filter f(q, q0)
			qty = filter(qty, qty_init)

			all_values[i].append(qty)
			if i == 0:
				all_x.append(x)
			all_t[i] = solver.time

	''' Compute global clim '''
	if clims is None:
		min_val = np.min([np.min([np.min(vals_dom) for vals_dom in vals_t]) for vals_t in all_values])
		max_val = np.max([np.max([np.max(vals_dom) for vals_dom in vals_t]) for vals_t in all_values])
		clims = (min_val, max_val,)

	''' Generate plot '''
	for i in range(num_frames):
		plt.clf()
		for dom_idx, name in enumerate(atm_names):
			plot_mean(all_x[dom_idx], all_values[i][dom_idx], clims)
		for dom_idx, name in enumerate(conduit_names):
			global_idx = dom_idx + len(atm_names)
			plot_mean1D(all_x[global_idx], all_values[i][global_idx], clims, xscale=50, xshift=0.0)
		
		sm = plt.cm.ScalarMappable(
			norm=matplotlib.colors.Normalize(vmin=clims[0], vmax=clims[1]),
			cmap=plt.get_cmap())
		cb = plt.colorbar(sm)
		cb.set_label(plot.get_ylabel(solver.physics, plot_qty, None))
		
		plt.title(f"t = {all_t[i]:.6f}")
		plt.axis("auto")
		plt.axis("equal")
		plt.pause(0.010)
		# plt.show()
		plt.draw()

		# Grab frame with FFwriter
		if not FFsetup_failure:
			FFwriter.grab_frame()

	if not FFsetup_failure:
		FFwriter.finish()
	print("Animation constructed")

def set_cbar(clims, cbar_title=None):
	cb = plt.colorbar(plt.cm.ScalarMappable(
				norm=matplotlib.colors.Normalize(vmin=clims[0], vmax=clims[1]),
				cmap=plt.get_cmap()))
	if cbar_title is not None:
		cb.set_label(cbar_title)

class SolverInterpolator():
	'''
	Interpolator over a solver's domain. Call with (x,y) to obtain state_coeffs
	at those locations. Interpolates using CubicTriInterpolator.
	'''
	def __init__(self, solver):
		self.solver = solver
		self.triangulation = tri.Triangulation(solver.mesh.node_coords[...,0],
									solver.mesh.node_coords[...,1], 
									triangles=solver.mesh.elem_to_node_IDs)
		# Trifinder (x,y) -> [elem_IDs]
		self.trifinder = self.triangulation.get_trifinder()
	
	def __call__(self, x:ArrayLike, y:ArrayLike) -> np.array:
		try:
			iter(x)
		except TypeError:
			x = np.array([x])
			y = np.array([y])
		solver = self.solver
		# Grab data to interpolate
		u = solver.state_coeffs
		if solver.basis.order == 1:
			out = np.full((np.array(x).ravel().shape[0], u.shape[-1]), np.nan)
			elem_IDs = self.trifinder(x, y)
			for i, elem_ID in enumerate(elem_IDs):
				if elem_ID != -1:
					# Use cubic tri interpolator using nodal values at mesh nodes
					out[i,:] = np.array([tri.CubicTriInterpolator(tri.Triangulation(
						*(solver.mesh.node_coords[
							solver.mesh.elem_to_node_IDs[elem_ID,:], :].T)),
						u[elem_ID, :, state_idx], kind="geom")(x[i],y[i])
						for state_idx in range(u.shape[-1])])
		else:
			print("Interpolator not called; solver order is not 1.")
		return out

class UnionInterpolator():
	'''
	Interpolator over several solvers' domains. Call with (x,y) to obtain
	state_coeffs at those locations. Interpolates using CubiTriInterpolator.
	'''
	def __init__(self, solvers):
		try:
			iter(solvers)
		except TypeError:
			solvers = [solvers]
		self.solvers = solvers
		self.interpolators = [SolverInterpolator(solver) for solver in solvers]
		self.mesh_centers = np.array([np.mean(solver.mesh.node_coords, axis=0)
			for solver in solvers])
		self.N_states = self.solvers[0].state_coeffs.shape[-1]
	
	def __call__(self, x:ArrayLike, y:ArrayLike) -> np.array:
		try:
			iter(x)
		except TypeError:
			x = np.array([x])
			y = np.array([y])
		out = np.full((np.array(x).ravel().shape[0], self.N_states), np.nan)
		for i, (x_pt, y_pt) in enumerate(zip(x, y)):
			# Set priority of solvers based on proximity to mesh center
			priority_range = np.argsort(
				np.linalg.norm(np.array([x_pt, y_pt]) - self.mesh_centers, axis=-1))
			for solver_idx in priority_range:
				interpolated_state = self.interpolators[solver_idx]([x_pt], [y_pt])
				if ~np.any(np.isnan(interpolated_state)):
					break
			out[i,:] = interpolated_state
		return out
	
	def get_sample_points(x:ArrayLike, y:ArrayLike, t:ArrayLike):
		# Parallel sampling for ArrayLike (x, y) in R^n_x , t in R^n_t
		pass

	def get_sample_gradients(x:ArrayLike, y:ArrayLike, t:ArrayLike):
		pass

def UnionProcessing():
	# Prototype; see XYTInterpolator below.

	import glob
	from pickle import UnpicklingError

	file_path = "../gravity_settling/tungSS_atm"

	N_DECIMALS_THRESHOLDING = 1

	# Convert file path with file radical to search string
	search_str = f"{file_path}*.pkl"

	''' Create file model and cache results. '''
	file_names = glob.glob(search_str)
	# Get list of initial meshes TODO: needs a mesh from somewhere even if _0 DNE
	initials = [fname for fname in file_names if fname[-6:] == "_0.pkl"]
	# Grab list of meshes
	meshes = [readwritedatafiles.read_data_file(fname).mesh for fname in initials]
	N_meshes = len(meshes)
	solvers_in_mesh = [[] for i in range(N_meshes)]
	timeline_in_mesh = [[] for i in range(N_meshes)]
	# Initialize mapping fname -> file_data
	data_rep = {}
	# Read physics from first encountered file
	physics = readwritedatafiles.read_data_file(file_names[0]).physics

	# Define block view (compressing (x,y) -> v)
	asblock = lambda x: x.ravel().view(np.dtype((np.void, 2*x.dtype.itemsize)))

	for fname in file_names:
		try:
			# Read file
			solver = readwritedatafiles.read_data_file(fname)
		except UnpicklingError:
			print(f"Skipping {fname}: unpicklable file.")
			continue
		# Match mesh by node coordinates (exact equality of floats) interpreted as
		# 2*float64 blocks
		mesh_index = np.argwhere(
			[len(np.union1d(asblock(solver.mesh.node_coords),
			asblock(mesh.node_coords))) == len(solver.mesh.node_coords)
			for mesh in meshes]).ravel()[0]
		# Save simplified representation of current file
		data_rep[fname] = ({
			"t": solver.time,
			"mesh_index": mesh_index,
			"state_coeffs": solver.state_coeffs,
		})
		# Save name in list solvers_in_mesh
		solvers_in_mesh[mesh_index].append(fname)

	''' Postprocess data entries for each mesh '''
	for i in range(N_meshes):
		# Sort files by increasing t in each mesh
		solvers_in_mesh[i].sort(key=lambda fname: data_rep[fname]["t"])
		# Remove duplicate solvers at each time
		timeline_in_mesh[i], argunique = np.unique([data_rep[fname]["t"]
			for fname in solvers_in_mesh[0]], return_index=True)
		solvers_in_mesh[i] = list(np.array(solvers_in_mesh[i])[argunique])

	''' Construct union trifinder '''
	# Compute shifts to assign unique node num to union of meshes
	index_shifts = np.roll(np.cumsum([mesh.num_nodes for mesh in meshes]),1)
	index_shifts[0] = 0
	# Construct mapping from element num (row) to ID in union of meshes
	union_elem_to_node_IDs = np.vstack([mesh.elem_to_node_IDs+shift
		for mesh, shift in zip(meshes, index_shifts)])
	# Construct global set of nodes, with duplicate nodes
	z = np.vstack(tuple(mesh.node_coords[...,:] for mesh in meshes))

	''' Get map from bdry key (name of interface bdry) to global node
	number. Useful if the nodes lying on boundaries need to be accessed in the
	union view. Currently only used along with argunique to verify rounding
	threshold is sufficient; the mapping from original node IDs to new node IDs,
	with duplicates removed, needs more information than argunique. '''
	bdry2nodes = {}
	for i in range(N_meshes):
		# Read initial output for mesh i
		solver = readwritedatafiles.read_data_file(solvers_in_mesh[i][0])
		mesh = solver.mesh
		# Find linked boundaries
		linked_bdrys = [key for key, val in solver.physics.BCs.items() if "2D2D" in str(val)]
		for bdry in linked_bdrys:
			# Get nodes on boundary in domain (elem_to_node_IDs accessed at [bdry elt
			# ID, element-local node ID])
			bdry_node_IDs = [ index_shifts[i] +
				mesh.elem_to_node_IDs[bdryface.elem_ID,
					mesh.gbasis.get_local_face_principal_node_nums(
					mesh.gorder,
					bdryface.face_ID)]
				for bdryface in mesh.boundary_groups[bdry].boundary_faces
				]
			bdry2nodes[bdry] = bdry2nodes.get(bdry, [])
			bdry2nodes[bdry].extend(np.unique(bdry_node_IDs))
	# Coordinate thresholding for approximate equality at boundaries
	z_round = np.round(z,decimals=N_DECIMALS_THRESHOLDING)
	# Get sort index for unique coordinates (up to rounding)
	_, argunique = np.unique(asblock(z_round), return_index=True)
	# Assert number of unique nodes is equal to number of nodes minus half of the
	# nodes with duplicated coordinates (i.e. num nodes on boundaries shared by
	# two domains).
	assert(len(z) - len(argunique) == 
		sum([len(bdry2nodes[key]) for key in bdry2nodes.keys()]) / 2)

	# Map from old node IDs (with duplicates) to new node IDs (without
	# duplicates and contiguous)
	map2new = np.zeros((len(z_round),), dtype=int)
	# Sorted sequence of multi-byte blocks
	sortidx = np.argsort(asblock(z_round))
	seq_sorted = asblock(z_round)[sortidx]
	# Construction of unique node coords
	unique_nodes = [z[sortidx][0]]

	# Construct mapping to contiguous unique coordinates
	k = 0
	for i in range(len(map2new)):
		if i >= 1 and seq_sorted[i] != seq_sorted[i-1]:
			k += 1
			unique_nodes.append(z[sortidx][i])
		map2new[i] = k

	unique_nodes = np.array(unique_nodes)
	# Migrate references to old node IDs to new node IDs in each triangle
	invsortidx = sortidx.argsort()
	union_elem_to_node_IDs_unique = map2new[invsortidx[union_elem_to_node_IDs]]

	global_trifinder = tri.Triangulation(*unique_nodes.T, 
		union_elem_to_node_IDs_unique).get_trifinder()
	pass


class XYTInterpolator():
	''' General ((x,y), t) interpolator. Performs linear interpolation in t.
	Automatic file ingestion for given file path + prefix. '''

	N_DECIMALS_THRESHOLDING = 1

	def __init__(self, file_path):
		# Convert file path with file radical to search string
		search_str = f"{file_path}*.pkl"

		''' Create data model and ingest data. '''
		file_names = glob.glob(search_str)
		# Get list of initial meshes TODO: needs a mesh from somewhere even if _0 DNE
		initials = [fname for fname in file_names if fname[-6:] == "_0.pkl"]
		meshes = [readwritedatafiles.read_data_file(fname).mesh for fname in initials]
		N_meshes = len(meshes)
		solvers_in_mesh = [[] for i in range(N_meshes)]
		timeline_in_mesh = [[] for i in range(N_meshes)]
		# Initialize mapping fname -> file_data
		data_rep = {}
		# Read physics from first encountered file
		physics = readwritedatafiles.read_data_file(file_names[0]).physics
		N_states = physics.NUM_STATE_VARS
		# Define block view (compressing (x,y) -> v)
		asblock = lambda x: x.ravel().view(np.dtype((np.void, 2*x.dtype.itemsize)))

		for fname in file_names:
			try:
				# Read file
				solver = readwritedatafiles.read_data_file(fname)
			except UnpicklingError:
				print(f"Skipping {fname}: unpicklable file.")
				continue
			if solver.basis.order == 1:
				''' This postprocessor is implemented for order == 1 for 2D triangular
				elements. TODO: generalize to other orders, where the nodes are not
				necessarily located at the triangle vertices. The latter assumption
				allows simple linear interpolation in 2D tris. '''
			else:
				raise NotImplementedError("Solver basis is not order 1.")
			# Match mesh by node coordinates (exact equality of floats) interpreted as
			# 2*float64 blocks
			mesh_index = np.argwhere(
				[len(np.union1d(asblock(solver.mesh.node_coords),
				asblock(mesh.node_coords))) == len(solver.mesh.node_coords)
				for mesh in meshes]).ravel()[0]
			# Save simplified representation of current file
			data_rep[fname] = ({
				"t": solver.time,
				"mesh_index": mesh_index,
				"state_coeffs": solver.state_coeffs,
			})
			# Save name in list solvers_in_mesh
			solvers_in_mesh[mesh_index].append(fname)

		''' Postprocess data entries for each mesh '''
		for i in range(N_meshes):
			# Sort files by increasing t in each mesh
			solvers_in_mesh[i].sort(key=lambda fname: data_rep[fname]["t"])
			# Remove duplicate solvers at each time
			timeline_in_mesh[i], argunique = np.unique([data_rep[fname]["t"]
				for fname in solvers_in_mesh[0]], return_index=True)
			solvers_in_mesh[i] = list(np.array(solvers_in_mesh[i])[argunique])

		''' Construct union trifinder '''
		# Compute shifts to assign unique node num to union of meshes
		index_shifts = np.roll(np.cumsum([mesh.num_nodes for mesh in meshes]),1)
		index_shifts[0] = 0
		# Construct mapping from element num (row) to ID in union of meshes
		union_elem_to_node_IDs = np.vstack([mesh.elem_to_node_IDs+shift
			for mesh, shift in zip(meshes, index_shifts)])
		# Construct global set of nodes, with duplicate nodes
		z = np.vstack(tuple(mesh.node_coords[...,:] for mesh in meshes))

		''' Get map from bdry key (name of interface bdry) to global node
		number. Useful if the nodes lying on boundaries need to be accessed in the
		union view. Currently only used along with argunique to verify rounding
		threshold is sufficient; the mapping from original node IDs to new node IDs,
		with duplicates removed, needs more information than argunique. '''
		bdry2nodes = {}
		for i in range(N_meshes):
			# Read initial output for mesh i
			solver = readwritedatafiles.read_data_file(solvers_in_mesh[i][0])
			mesh = solver.mesh
			# Find linked boundaries
			linked_bdrys = [key for key, val in solver.physics.BCs.items()
				if "2D2D" in str(val)]
			for bdry in linked_bdrys:
				# Get nodes on boundary in domain (elem_to_node_IDs accessed at [bdry elt
				# ID, element-local node ID])
				bdry_node_IDs = [ index_shifts[i] +
					mesh.elem_to_node_IDs[bdryface.elem_ID,
						mesh.gbasis.get_local_face_principal_node_nums(
						mesh.gorder,
						bdryface.face_ID)]
					for bdryface in mesh.boundary_groups[bdry].boundary_faces
					]
				bdry2nodes[bdry] = bdry2nodes.get(bdry, [])
				bdry2nodes[bdry].extend(np.unique(bdry_node_IDs))
		# Coordinate thresholding for approximate equality at boundaries
		z_round = np.round(z,decimals=XYTInterpolator.N_DECIMALS_THRESHOLDING)
		# Get sort index for unique coordinates (up to rounding)
		_, argunique = np.unique(asblock(z_round), return_index=True)
		# Assert number of unique nodes is equal to number of nodes minus half of the
		# nodes with duplicated coordinates (i.e. num nodes on boundaries shared by
		# two domains).
		assert(len(z) - len(argunique) == 
			sum([len(bdry2nodes[key]) for key in bdry2nodes.keys()]) / 2)

		''' Construct global triangle finder. '''
		# Map from old node IDs (with duplicates) to new node IDs (without
		# duplicates and contiguous)
		map2new = np.zeros((len(z_round),), dtype=int)
		# Sorted sequence of multi-byte blocks
		sortidx = np.argsort(asblock(z_round))
		seq_sorted = asblock(z_round)[sortidx]
		# Construction of unique node coords
		unique_nodes = [z[sortidx][0]]

		# Construct mapping to contiguous unique coordinates
		k = 0
		for i in range(len(map2new)):
			if i >= 1 and seq_sorted[i] != seq_sorted[i-1]:
				k += 1
				unique_nodes.append(z[sortidx][i])
			map2new[i] = k

		unique_nodes = np.array(unique_nodes)
		# Migrate references to old node IDs to new node IDs in each triangle
		invsortidx = sortidx.argsort()
		union_elem_to_node_IDs_unique = map2new[invsortidx[union_elem_to_node_IDs]]
		# Construct global trifinder
		global_trifinder = tri.Triangulation(*unique_nodes.T, 
			union_elem_to_node_IDs_unique).get_trifinder()

		''' Perform other initialization tasks needed by methods. '''
		# Compute cumulative counter for elements
		#   (N < elt_count[i] and (i == 0 or N > elt_count[i-1]) in element i)
		elt_count = np.cumsum([mesh.elem_to_node_IDs.shape[0] for mesh in meshes])
		# Construct domain-specific triangulation
		local_triangulations = [tri.Triangulation(mesh.node_coords[...,0],
			mesh.node_coords[...,1], 
			triangles=mesh.elem_to_node_IDs) for mesh in meshes]

		''' Export variables '''
		self.physics = physics
		self.data_model = {
			"N_meshes": N_meshes,
			"meshes": meshes,
			"solvers_in_mesh": solvers_in_mesh,
			"timeline_in_mesh": timeline_in_mesh,
			"data_rep" : data_rep,
			"local_triangulations": local_triangulations,
		}
		self.global_trifinder = global_trifinder
		self.elt_count = elt_count

	def domain_identifier(self, x:float, y:float) -> int:
		''' Mapping from x, y -> domain id (mesh/domain index in internal data
		model). '''
		dom_idx = self.global_trifinder(x,y)
		if dom_idx == -1:
			return -1
		else:
			return np.where(self.global_trifinder(x,y) < self.elt_count)[0][0]

	def __call__(self, x:np.array, y:np.array, t:np.array) -> np.array:
		''' Dispatch x, y to each mesh. For each domain, interpolate in t, and then
		interpolate in space. '''
		data_rep = self.data_model["data_rep"]
		solvers_in_mesh = self.data_model["solvers_in_mesh"]
		timeline_in_mesh = self.data_model["timeline_in_mesh"]
		meshes = self.data_model["meshes"]
		local_triangulations = self.data_model["local_triangulations"]
		N_meshes = self.data_model["N_meshes"]
		N_states = self.physics.NUM_STATE_VARS
		# Construct list of references to data on each mesh, indexed by time
		state_coeffs_global_time = [[data_rep[sname]["state_coeffs"] 
			for sname in solvers_in_mesh[i]]
			for i in range(N_meshes)]
		# Identify domain with internal index
		domain_dispatch_idx = [self.domain_identifier(x_,y_) for x_,y_ in zip(x,y)]
		# Set mapping for dispatching each (x, y) to a mesh index:
		#   list(tuple(index, point:tuple))
		dispatch_map = [[] for dom_idx in range(N_meshes)]
		for i, (x_,y_) in enumerate(zip(x,y)):
			dispatch_map[self.domain_identifier(x_,y_)].append((i,(x_,y_)))
		# Allocate output (len(t), len(x_), num_states)
		out = np.full((t.shape[0], np.array(x).ravel().shape[0], N_states), np.nan)
		for dom_idx, mesh_workload in enumerate(dispatch_map):
			''' Perform interpolation on each mesh. '''
			if len(mesh_workload) == 0:
				continue
			# Get indices for placing results (i_loc) and point geometry x_loc, y_loc
			i_loc, x_loc = list(zip(*mesh_workload))
			x_loc, y_loc = list(zip(*x_loc))
			# Compute interpolated time-indices as floats
			t_index_interp = (np.interp(t,
				timeline_in_mesh[dom_idx], np.array(range(len(timeline_in_mesh[dom_idx])))))
			# Mark invalid t interpolate requests (do not extrapolate)
			t_index_interp[t > timeline_in_mesh[dom_idx].max()] = np.nan
			t_index_interp[t < timeline_in_mesh[dom_idx].min()] = np.nan
			# Pull data
			for j, t_index in enumerate(t_index_interp):
				if not np.isnan(t_index):
					l = np.floor(t_index).astype(int)
					u = np.ceil(t_index).astype(int)
					theta = t_index - l
					assert(theta == np.clip(theta,0,1))
					# Linear interpolate in time the state coeffs over the domain
					U = (1-theta) * state_coeffs_global_time[dom_idx][l] \
						+ theta * state_coeffs_global_time[dom_idx][u]
					# Compute subset x
					elem_IDs = local_triangulations[dom_idx].get_trifinder()(x_loc, y_loc)
					for i, elem_ID in enumerate(elem_IDs):
						if elem_ID != -1:
							# Use cubic tri interpolator using nodal values at mesh nodes
							out[j,i_loc[i],:] = np.array([tri.CubicTriInterpolator(tri.Triangulation(
								*(meshes[dom_idx].node_coords[
									meshes[dom_idx].elem_to_node_IDs[elem_ID,:], :].T)),
								U[elem_ID, :, state_idx], kind="geom")(x_loc[i],y_loc[i])
								for state_idx in range(N_states)])
		return out
