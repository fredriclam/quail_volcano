  # Multidomain postprocessing toolkit
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import processing.readwritedatafiles as readwritedatafiles
import processing.plot as plot

def viz(solver, plot_qty:str="Pressure"):
  x, qty = downsample(solver, plot_qty)
  if solver.physics.NDIMS == 2:
    custom_plot_2D(x, qty, solver, levels=None)
  elif solver.physics.NDIMS == 1:
    plot(x.ravel(), qty.ravel(), '.-')

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

def custom_plot_2D(x, var_plot, solver, levels=None):
	# plot.plot_2D_general(solver.physics, x, var_plot)

	if levels is None:
		''' Open temp figure to get default levels'''
		figtmp = plt.figure()
		# Run this for the sole purpose of getting default contour levels
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
			[x[i,2,0], x[i,2,1]]], facecolor=cmap(cnorm(q[i,:,0].mean())))
		plt.gca().add_patch(pp)
	plt.axis("auto")
	plt.axis("equal")

def plot_mean1D(x, q, clims, xscale=1.0, xshift=0.0):
	''' Create very cheap plot '''
	cmap = plt.get_cmap()
	cnorm = matplotlib.colors.Normalize(vmin=clims[0], vmax=clims[1]) 
	for i in range(x.shape[0]):
		pp = plt.Rectangle((xshift, x[i,0,0]), width=xscale, height=x[i,1,0]-x[i,0,0],
			facecolor=cmap(cnorm(q[i,:,0].mean())))
		plt.gca().add_patch(pp)
	plt.axis("auto")
	plt.axis("equal")

def generate_anim(atm_names, conduit_names, outfilename, num_frames,
  plot_qty, filter=lambda x, x0:x, initial=1, stride=10, is_high_detail=False):
  plt.rcParams['animation.ffmpeg_path'] = "C:\\Users\\Fredric\\Documents\\ffmpeg\\ffmpeg-n4.4-latest-win64-gpl-4.4\\bin\\ffmpeg.exe"
  fig = plt.figure()

  FFwriter = animation.FFMpegWriter()
  FFwriter.setup(fig, f"{outfilename}.mp4")

  atm_initials = [readwritedatafiles.read_data_file(f"{name}_{0}.pkl") 
    for name in atm_names]
  conduit_initials = [readwritedatafiles.read_data_file(f"{name}_{0}.pkl") 
    for name in conduit_names]

  ''' Compute values for each frame '''
  all_values = []
  all_x = []
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
      if i == 1:
        all_x.append(x)


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
      if i == 1:
        all_x.append(x)

  ''' Compute global clim '''
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
    
    plt.axis("auto")
    plt.axis("equal")
    plt.pause(0.010)
    # Grab frame with FFwriter
    FFwriter.grab_frame()

  FFwriter.finish()
  print("Animation constructed")