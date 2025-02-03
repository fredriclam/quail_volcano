# ------------------------------------------------------------------------ #
#
#       File : src/processing/animate.py
#
#       Contains functions for animating various plots.
#
# ------------------------------------------------------------------------ #
from matplotlib import pyplot as plt
import numpy as np
import os

import meshing.tools as mesh_tools

from matplotlib import animation
from processing import readwritedatafiles

def test_function():
	print("Hello world!")
	return "test"


def animate_conduit_pressure(folder, iterations=100, file_prefix="test_output", viscosity_index=0):
	"""This function takes in a folder, file prefix, and number of iterations and returns an animation of various state variables in the conduit over time.
	
	Parameters
	----------
	folder (str): The folder containing the data files.
	iterations (int): The number of iterations in the simulation.
	fil	_prefix (str): The prefix of the data files.
	viscosity_index (int): The index of the viscosity source term in the source terms list.
	"""

	fig = plt.figure(figsize=(10,8))
	ax = fig.add_subplot(321,autoscale_on=False,\
                            xlim=(0,-1000),ylim=(0,3))
	ax2 = fig.add_subplot(322,autoscale_on=False,\
                            xlim=(0,-1000),ylim=(-0.5,1.5))
	ax3 = fig.add_subplot(323, autoscale_on=False,\
                            xlim=(0,-1000), ylim=(0,1200))
	ax4 = fig.add_subplot(324, autoscale_on=False,\
                            xlim=(0,-1000), ylim=(0,1))
	ax5 = fig.add_subplot(325, autoscale_on=False,\
                            xlim=(0,-1000), ylim=(-0.02,0.02)) 
	ax6 = fig.add_subplot(326, autoscale_on=False, \
							xlim=(0,-1000), ylim=(-0.1,0.1))

	pressure_line,  = ax.plot([], [], color="blue", label="pressure")
	velocity_line, = ax2.plot([], [], color="red", label="velocity")
	sound_speed_line, = ax3.plot([], [], color="green", label="speed of sound")
	viscosity_line, = ax4.plot([], [], color="orange", label="viscosity")


	total_water_line, = ax5.plot([], [], color="purple", label="total water")
	exsolved_water_line, = ax5.plot([], [], color="blue", label="exsolved water")

	new_state_line, = ax6.plot([], [], color="purple", label="new state")

	ax5.legend(loc="upper right")
	ax5.set_xlabel("Depth [m]")
	ax6.set_xlabel("Depth [m]")

	ax.set_ylabel("Pressure [MPa]")
	ax2.set_ylabel("Velocity [m/s]")
	ax3.set_ylabel("Speed of sound [m/s]")
	ax4.set_ylabel("Effective viscosity [MPa * s]")
	ax5.set_ylabel("Water partial density")

	time_template = 'time = %.2f [s]'
	time_text = ax.text(0.5,0.9,'',transform=ax.transAxes)

	pl_template = 'P_L = %2f [M Pa]'
	pl_text = ax.text(0.5, 0.8, "", transform=ax.transAxes)

	velocity_template = 'V = %2f [m/s]'
	velocity_text = ax2.text(0.5, 0.9, "", transform=ax2.transAxes)

	print(os.getcwd())

	def init():
		pressure_line.set_data([], [])
		velocity_line.set_data([], [])
		sound_speed_line.set_data([], [])
		viscosity_line.set_data([], [])
		total_water_line.set_data([], [])
		exsolved_water_line.set_data([], [])
		new_state_line.set_data([], [])
	
		time_text.set_text("")
		pl_text.set_text("")
		velocity_text.set_text("")
		return pressure_line, velocity_line, viscosity_line, total_water_line, exsolved_water_line, new_state_line, time_text, pl_text, velocity_text

	def animate(i):
		solver = readwritedatafiles.read_data_file(f"{folder}/{file_prefix}_{i}.pkl")
		flag_non_physical = True
		p = solver.physics.compute_additional_variable("Pressure", solver.state_coeffs, flag_non_physical)
		v = solver.physics.compute_additional_variable("Velocity", solver.state_coeffs, flag_non_physical)
		sound_speed = solver.physics.compute_additional_variable("SoundSpeed", solver.state_coeffs, flag_non_physical)

		fsource = solver.physics.source_terms[viscosity_index]
		viscosity = fsource.compute_viscosity(solver.state_coeffs, solver.physics)

		arhoWt = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityWt")]
		arhoWv = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityWv")]

		# Get the value of the new state variable.
		arhoX = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityX")]

		# Get the position of of each nodal points (location corresponding to each entry of pDensityX)
		nodal_pts = solver.basis.get_nodes(solver.order)
		# Allocate [ne] x [nb, ndims]
		x = np.empty((solver.mesh.num_elems,) + nodal_pts.shape)
		for elem_ID in range(solver.mesh.num_elems):
			# Fill coordinates in physical space
			x[elem_ID] = mesh_tools.ref_to_phys(solver.mesh, elem_ID, nodal_pts)
	
		pressure_line.set_data(x.ravel(), p.ravel()/1e6)
		velocity_line.set_data(x.ravel(), v.ravel())
		sound_speed_line.set_data(x.ravel(), sound_speed.ravel())
		viscosity_line.set_data(x.ravel(), viscosity.ravel()/1e6)
		total_water_line.set_data(x.ravel(), arhoWt.ravel())
		exsolved_water_line.set_data(x.ravel(), arhoWv.ravel())
		new_state_line.set_data(x.ravel(), arhoX.ravel())

		time_text.set_text(time_template % solver.time)
		pl_text.set_text(pl_template % (p.ravel()/1e6)[0])
		velocity_text.set_text(velocity_template % (v.ravel())[0])

		return pressure_line, velocity_line, sound_speed_line, viscosity_line, total_water_line, exsolved_water_line, new_state_line, time_text, pl_text, velocity_text

	plt.close()
	return animation.FuncAnimation(fig, animate, np.arange(iterations), blit=False, init_func=init, interval=40)