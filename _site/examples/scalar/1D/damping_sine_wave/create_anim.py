import code

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles


plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = plt.gca()


imgs_all = []
j = 0
for i in range(16):
	print(i)
	fname = "Data_" + str(i) + ".pkl"
	solver = readwritedatafiles.read_data_file(fname)
	# Unpack
	mesh = solver.mesh
	physics = solver.physics

	# plot.PlotSolution(mesh, physics, solver, "Scalar", create_new_figure=False, PlotExact=True, PlotIC=True, Label="u",
	# 		ignore_legend=True)
	plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True, plot_exact=False, plot_IC=False, create_new_figure=False, 
			ylabel=None, fmt='bo', legend_label="DG", equidistant_pts=True, 
			include_mesh=False, regular_2D=False, equal_AR=False, ignore_legend=True)
	plot.plot_solution(mesh, physics, solver, "Scalar", plot_exact=True, plot_numerical=False, create_new_figure=False, 
			fmt='k-', ignore_legend=True)
	plot.plot_solution(mesh, physics, solver, "Scalar", plot_IC=True, plot_numerical=False, create_new_figure=False, 
			fmt='k--', ignore_legend=True)

	imgs = ax.get_lines().copy()

	if j == 0:
		plt.legend(loc="best")
		imgs_all.append(imgs)
	else:
		nc = len(imgs_all[j-1])
		imgs_all.append(imgs[-nc:])

	j += 1


anim = animation.ArtistAnimation(fig, imgs_all, interval=500, blit=False,
                                repeat_delay=None)

plt.show()

anim.save("anim.mp4")
