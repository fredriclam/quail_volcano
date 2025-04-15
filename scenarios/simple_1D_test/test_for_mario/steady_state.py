# Specify quail source directory
# Modify base path for depending on your file structure.
BASE_PATH = "/Users/paxton/git"

# Specify path for Quail source code
source_dir = f"{BASE_PATH}/quail_volcano/src"

# Import standard libraries
import matplotlib.pyplot as plt
import numpy as np
import os
# Import steady_state module
os.chdir(source_dir)
print(os.getcwd())
import compressible_conduit_steady.steady_state as steady_state

# Define plug parameters
radio = 5
f_plug = 1.9e8
len_plug = 50

t_plug = f_plug/(2*np.pi*radio*len_plug)
trac_par = 2*t_plug/radio

# Construct x_mesh (with shape (n, 1, 1))
N_mesh_points = 800

x_mesh = np.linspace(-1000, 0, N_mesh_points)[:,np.newaxis,np.newaxis]
# Chamber pressure is not used, so we pass a dummy value
p_chamber = None
# Set vent pressure

p_vent = (1e5)

# Set functions for traction, total water mass fraction, crystal mass fraction, temperature
# Define the cosine taper function
def cosine_taper(x, x1, x2, y1, y2):
    return np.where(x < x1, y1,
                    np.where(x > x2, y2,
                             y1 + (y2 - y1) * 0.5 * (1 - np.cos(np.pi * (x - x1) / (x2 - x1)))))

# Define the transition region
x1 = -len_plug - 10  # Start of transition
x2 = -len_plug + 10  # End of transition

# Define the functions using cosine taper
traction_fn = lambda x: cosine_taper(x, x1, x2, 0, -trac_par)
yWt_fn = lambda x: cosine_taper(x, x1, x2, 0.04, 0.01)
yC_fn = lambda x: cosine_taper(x, x1, x2, 0.4, 0.95)
T_fn = lambda x: cosine_taper(x, x1, x2, 950 + 273.15, 930 + 273.15)
yF_fn = lambda x: cosine_taper(x, x1, x2, 0, 1)

# Set material properties of the magma phase (melt + dissolved water + crystals)
material_props = {
  "yA": 1e-7,          # Air mass fraction (> 0 for numerics)
  "c_v_magma": 1e3,    # Magma phase heat capacity per mass
  "rho0_magma": 2.6e3, # Linearization reference density
  "K_magma": 10e9,     # Bulk modulus
  "p0_magma": 36e6,    # Linearization reference pressure
  "solubility_k": 2.8e-6, # Henry's law coefficient
  "solubility_n": 0.5, # Henry's law exponent
}

# Initialize hydrostatic steady-state solver
# This is a one-use callable object
f = steady_state.StaticPlug(x_mesh,
                            p_chamber,
                            traction_fn, yWt_fn, yC_fn, T_fn, yF_fn,
                            override_properties=material_props, enforce_p_vent=p_vent)
# Solve by calling f
#   io_format="p" here returns only pressure
#   io_format="quail" will return the solution in quail format
p = f(x_mesh, is_solve_direction_downward=True, io_format="p")

# Solve again in Quail format (need to reinitialize f)
f = steady_state.StaticPlug(x_mesh,
                            p_chamber,
                            traction_fn, yWt_fn, yC_fn, T_fn, yF_fn,
                            override_properties=material_props, enforce_p_vent=p_vent)
U = f(x_mesh, is_solve_direction_downward=True, io_format="quail")


#Plots
plt.figure()
plt.plot( yWt_fn(x_mesh).ravel(), (x_mesh.ravel()))
plt.xlabel('Water content',  fontsize=13)
#plt.ylim((-700, -500))

# ravel() smooshes the dimension of the data into a 1-D vector for plotting
plt.figure(figsize=(6,8))
plt.subplot(3,1,1)
plt.plot(x_mesh.ravel(), p.ravel())

plt.xlabel("x (m)")
plt.ylabel("p (Pa)")

# Here we work with the quail-formatted data
plt.subplot(3,1,2)
rho = U[...,0:3].sum(axis=-1, keepdims=True)
plt.plot(x_mesh.ravel(), rho.ravel())
plt.xlabel("x (m)")
plt.ylabel("Mixture density (kg/m^3)")


plt.subplot(3,1,3)
plt.plot(x_mesh.ravel(), (U[...,1:2]/rho).ravel())
plt.xlabel("x (m)")
plt.ylabel("Mass fraction of exsolved water")
plt.tight_layout()