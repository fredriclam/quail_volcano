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


def animate_conduit_pressure(folder, iterations=100, file_prefix="test_output", viscosity_index=2, wall_friction_index=1, plug_boundary=25, p0=4.1, x_min=-1000, max_speed_of_sound=1000, max_pressure=12, max_velocity=1, slip_final = 3.5, max_slip=20, max_tau=0.5):
	"""This function takes in a folder, file prefix, and number of iterations and returns an animation of various state variables in the conduit over time.
	
	Parameters
	----------
	folder (str): The folder containing the data files.
	iterations (int): The number of iterations in the simulation.
	fil	_prefix (str): The prefix of the data files.
	viscosity_index (int): The index of the viscosity source term in the source terms list.
	"""

	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(521,autoscale_on=False,\
                            xlim=(0,x_min),ylim=(0,max_pressure))
	ax2 = fig.add_subplot(522,autoscale_on=False,\
                            xlim=(0,x_min),ylim=(-0.2,max_velocity))
	ax3 = fig.add_subplot(523, autoscale_on=False,\
                            xlim=(0,x_min), ylim=(0,max_speed_of_sound))
	ax4 = fig.add_subplot(524, autoscale_on=False,\
                            xlim=(0,x_min), ylim=(0,1.2))
	ax5 = fig.add_subplot(525, autoscale_on=False,\
                            xlim=(0,x_min), ylim=(-0.02,0.02)) 
	ax6 = fig.add_subplot(526, autoscale_on=False, \
							xlim=(0,x_min), ylim=(-1,max_slip))
	ax7 = fig.add_subplot(527, autoscale_on=False, \
							xlim=(0,x_min), ylim=(0,2000))
	ax8 = fig.add_subplot(528, autoscale_on=False, \
							xlim=(0,x_min), ylim=(-1,150))
	ax9	= fig.add_subplot(529, autoscale_on=False, \
							xlim=(0,x_min), ylim=(2400,2800))	
	ax10 = fig.add_subplot(5,2,10, autoscale_on=False, \
							xlim=(0,-500), ylim=(-0.1,max_tau))

	ax.invert_xaxis()
	ax2.invert_xaxis()
	ax3.invert_xaxis()
	ax4.invert_xaxis()
	ax5.invert_xaxis()
	ax6.invert_xaxis()
	ax7.invert_xaxis()
	ax8.invert_xaxis()
	ax9.invert_xaxis()
	ax10.invert_xaxis()

	pressure_line,  = ax.plot([], [], color="blue", label="pressure")
	velocity_line, = ax2.plot([], [], color="red", label="velocity")
	#analytical_velocity_line, = ax2.plot([], [], color="blue", label="analytical velocity")
	sound_speed_line, = ax3.plot([], [], color="green", label="speed of sound")
	viscosity_line, = ax4.plot([], [], color="orange", label="viscosity")
	total_water_line, = ax5.plot([], [], color="purple", label="total water")
	exsolved_water_line, = ax5.plot([], [], color="blue", label="exsolved water")
	new_state_line, = ax6.plot([], [], color="purple", label="new state")
	temp_line, = ax7.plot([], [], label="temperature")
	crystal_line, = ax8.plot([], [], label="crystals")
	plug_boundary_line = ax10.axvline(x=-500, color='b', linestyle='--', linewidth=1, label="plug boundary")
	arhoM_line, = ax9.plot([], [], label="melt")
	tau_line, = ax10.plot([], [], label="tau slip")
	tau_viscous_line, = ax10.plot([], [], label="tau viscous")

	ax2.axhline(y=0, color='k', linestyle='dashed')
	ax6.axhline(y=slip_final, color='k', linestyle='dashed', label="predicted slip")
	ax.axhline(y=p0, color='k', linestyle='dashed', label="predicted p0")

	#ax2.legend(loc="upper right")
	ax5.legend(loc="upper right")
	ax10.legend(loc="upper left")

	ax5.set_xlabel("Depth [m]")
	ax6.set_xlabel("Depth [m]")

	ax.set_ylabel("Pressure [MPa]")
	ax2.set_ylabel("Velocity [m/s]")
	ax3.set_ylabel("Speed of sound [m/s]")
	ax4.set_ylabel("Effective viscosity [MPa * s]")
	ax5.set_ylabel("Water partial density")
	ax6.set_ylabel("Slip (slip / rho_mix) [m]")
	ax7.set_ylabel("Temperature [K]")
	ax8.set_ylabel("Crystal partial density [kg/m^3]")
	ax9.set_ylabel("Melt partial density [kg/m^3]")
	ax10.set_ylabel("Tau [MPa]")

	time_template = 'time = %.2f [s]'
	time_text = ax.text(0.5,0.9,'',transform=ax.transAxes)

	pl_template = 'P_L = %2f [M Pa]'
	pl_text = ax.text(0.5, 0.8, "", transform=ax.transAxes)

	pr_template = 'P_R = %2f [M Pa]'
	pr_text = ax.text(0.5, 0.7, "", transform=ax.transAxes)

	velocity_template = 'V = %2f [m/s]'
	velocity_text = ax2.text(0.5, 0.9, "", transform=ax2.transAxes)

	boundary_template = 'Boundary = %2f'
	boundary_text = ax10.text(0.5, 0.9, "", transform=ax10.transAxes)

	tau_slip_template = "Tau slip = %2f [MPa]"
	tau_slip_text = ax10.text(0.5, 0.8, "", transform=ax10.transAxes)

	print(os.getcwd())

	def init():
		pressure_line.set_data([], [])
		velocity_line.set_data([], [])
		#analytical_velocity_line.set_data([], [])
		sound_speed_line.set_data([], [])
		viscosity_line.set_data([], [])
		total_water_line.set_data([], [])
		exsolved_water_line.set_data([], [])
		new_state_line.set_data([], [])
		temp_line.set_data([], [])
		crystal_line.set_data([], [])
		arhoM_line.set_data([], [])
		tau_line.set_data([], [])
		tau_viscous_line.set_data([], [])
	
		time_text.set_text("")
		pl_text.set_text("")
		pr_text.set_text("")
		velocity_text.set_text("")
		boundary_text.set_text("")
		tau_slip_text.set_text("")

		return pressure_line, velocity_line, viscosity_line, total_water_line, exsolved_water_line, new_state_line, temp_line, crystal_line, plug_boundary_line, arhoM_line, tau_line, tau_viscous_line, time_text, pl_text, pr_text, velocity_text, boundary_text, tau_slip_text

	def animate(i):
		solver = readwritedatafiles.read_data_file(f"{folder}/{file_prefix}_{i}.pkl")
		flag_non_physical = True
		p = solver.physics.compute_additional_variable("Pressure", solver.state_coeffs, flag_non_physical)
		sound_speed = solver.physics.compute_additional_variable("SoundSpeed", solver.state_coeffs, flag_non_physical)
		temp = solver.physics.compute_additional_variable("Temperature", solver.state_coeffs, flag_non_physical)

		fsource = solver.physics.source_terms[viscosity_index]
		viscosity = fsource.compute_viscosity(solver.state_coeffs, solver.physics)

		# Get the position of of each nodal points (location corresponding to each entry of slip)
		nodal_pts = solver.basis.get_nodes(solver.order)
		# Allocate [ne] x [nb, ndims]
		x = np.empty((solver.mesh.num_elems,) + nodal_pts.shape)

		for elem_ID in range(solver.mesh.num_elems):
			# Fill coordinates in physical space
			x[elem_ID] = mesh_tools.ref_to_phys(solver.mesh, elem_ID, nodal_pts)

		fsource_tau = solver.physics.source_terms[wall_friction_index]
		tau_slip = fsource_tau.compute_tau(solver.state_coeffs, x, solver.physics)
		plug_boundary = fsource_tau.compute_plug_boundary(solver.state_coeffs, x, solver.physics)

		tau_viscous = fsource.compute_tau(solver.state_coeffs, x, solver.physics)

		arhoA = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityA")]
		arhoWt = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityWt")]
		arhoWv = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityWv")]
		arhoC = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityC")]
		arhoM = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityM")]
		arhoF = solver.state_coeffs[:,:,solver.physics.get_state_index("pDensityFm")]

		# Get the value of the new state variable.
		rhoSlip = solver.state_coeffs[:,:,solver.physics.get_state_index("rhoSlip")]

		momentum = solver.state_coeffs[:,:,solver.physics.get_momentum_slice()]
		rho = np.sum(solver.state_coeffs[:, :, solver.physics.get_mass_slice()],axis=2,keepdims=True)

		# Define velocity as momentum divided by density. "velocity" when computed as an additional state variable appears to be an absolute value. 
		u = momentum.ravel() / rho.ravel()

		slip = rhoSlip.ravel() / rho.ravel()

		tau = 5e4 - (5e4 - 2e5)*np.exp(-slip/10)
		#analytical_velocity = (50**2 / (8 * viscosity)) * ((10e6 - 1e6) / 1000 - 2 *tau / 50)

		pressure_line.set_data(x.ravel(), p.ravel()/1e6)
		velocity_line.set_data(x.ravel(), u.ravel())
		#analytical_velocity_line.set_data(x.ravel(), analytical_velocity.ravel())
		sound_speed_line.set_data(x.ravel(), sound_speed.ravel())
		viscosity_line.set_data(x.ravel(), viscosity.ravel()/1e6)
		total_water_line.set_data(x.ravel(), arhoWt.ravel())
		exsolved_water_line.set_data(x.ravel(), arhoWv.ravel())
		new_state_line.set_data(x.ravel(), slip.ravel())
		temp_line.set_data(x.ravel(), temp.ravel())
		crystal_line.set_data(x.ravel(), arhoC.ravel())
		plug_boundary_line.set_xdata([plug_boundary])
		arhoM_line.set_data(x.ravel(), arhoM.ravel())
		tau_line.set_data(x.ravel(), tau_slip.ravel()/1e6)
		tau_viscous_line.set_data(x.ravel(), tau_viscous.ravel()/1e6)
		time_text.set_text(time_template % solver.time)
		pl_text.set_text(pl_template % (p.ravel()/1e6)[0])
		pr_text.set_text(pr_template % (p.ravel()/1e6)[-1])
		velocity_text.set_text(velocity_template % u[0])
		boundary_text.set_text(boundary_template % (plug_boundary))
		tau_slip_text.set_text(tau_slip_template % (tau_slip.ravel()[-1]/1e6))

		return pressure_line, velocity_line, sound_speed_line, viscosity_line, total_water_line, exsolved_water_line, new_state_line, temp_line, crystal_line, plug_boundary_line, tau_line, tau_viscous_line, arhoM_line, time_text, pl_text, pr_text, velocity_text, boundary_text, tau_slip_text

	plt.close()
	return animation.FuncAnimation(fig, animate, np.arange(iterations), blit=False, init_func=init, interval=40)