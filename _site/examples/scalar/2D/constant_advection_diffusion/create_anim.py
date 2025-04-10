import matplotlib.animation as animation
import matplotlib.pyplot as plt

import processing.post as post
import processing.plot as plot
import processing.readwritedatafiles as readwritedatafiles
import numpy as np

plot.prepare_plot(linewidth=0.5)
fig = plt.figure()
ax = plt.gca()

imgs_all = []
j = 0
# Loop through data files
for i in range(33):
	print(i)

	# Read data file
	fname = "Data_" + str(i) + ".pkl"
	solver = readwritedatafiles.read_data_file(fname)

	# Unpack
	mesh = solver.mesh
	physics = solver.physics

	# Plot solution
	if j == 0:
		ignore_colorbar = False
	else:
		ignore_colorbar = True

	levels = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.25])

	plot.plot_solution(mesh, physics, solver, "Scalar", plot_numerical=True, create_new_figure=False, 
			include_mesh=True, regular_2D=True, equal_AR=False, show_elem_IDs=False, ignore_colorbar=ignore_colorbar,
			levels=levels)

	imgs = ax.collections.copy()

	# Add to imgs_all
	if j == 0:
		imgs_all.append(imgs)
	else:
		nc = len(imgs_all[j-1])
		imgs_all.append(imgs[-nc:])

	j += 1

anim = animation.ArtistAnimation(fig, imgs_all, interval=50, blit=False,
		repeat_delay=None)

plt.show()

# Save mp4
anim.save("anim.mp4")
