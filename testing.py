import copy
import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sci
from pyXSteam.XSteam import XSteam

#%%capture
#%run "/Users/Emily/Documents/GitHub/quail_volcano/src/quail" "conduit.py"

import processing.readwritedatafiles as readwritedatafiles

# Specify path where .pkl files are located
target_dir = "/Users/Emily/Documents/GitHub/quail_volcano/scenarios/conduit1D/"
# Specify path for Quail source code
source_dir = "/Users/Emily/Documents/GitHub/quail_volcano/src/"
# Change to working directory
os.chdir(target_dir)

def main():
   solution = sci.optimize.fsolve(f, (10 * 10, 99))
   print(solution)

def f(variables):
   (h, p) = variables

   # setting up pkl to get values
   file_idx = 800
   solver = readwritedatafiles.read_data_file(f"conduit1D_{file_idx}.pkl")

   # getting the necessary values
   pDensityWv = solver.state_coeffs[:, :, solver.physics.get_state_index("pDensityWv")][0]
   pDensityM = solver.state_coeffs[:, :, solver.physics.get_state_index("pDensityM")][0]
   x_momentum = solver.state_coeffs[:, :, solver.physics.get_state_index("XMomentum")][0]
   energy_density = solver.state_coeffs[:, :, solver.physics.get_state_index("Energy")][0]

   # constants from physics
   K = solver.physics.Liquid["K"]
   p_0 = solver.physics.Liquid["p0"]
   rho_0 = solver.physics.Liquid["rho0"]

   # importing PySteam
   steam_table = XSteam(XSteam.UNIT_SYSTEM_BARE)  # m/kg/sec/K/MPa/W

   # creating equations
   alpha_w = (p_0 - K - p + (K / rho_0) * pDensityM) / \
             (p_0 + K - p)
   internal_energy_magma = pDensityM / (1 - alpha_w)

   eqn_1 = ((x_momentum ** 2) / (2 * (pDensityWv + pDensityM))) \
           + pDensityWv * steam_table.u_ph(p, h) + pDensityM * internal_energy_magma - energy_density

   eqn_2 = -h + steam_table.u_ph(p, h) + (p * alpha_w / pDensityWv)
   return np.array([eqn_1, eqn_2]).ravel()


if __name__ == '__main__':
  main()