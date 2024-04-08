# Mach disk height history utility

import numpy as np

import meshing
import numerics
import physics.multiphasevpT.multiphasevpT
import processing
import processing.readwritedatafiles as readwritedatafiles
import processing.mdtools
import solver
import sys

import multiprocessing as mp
import multidomain

from time import perf_counter

t0 = perf_counter()

_this_script_name = sys.argv[0]
output_prefix = sys.argv[1]         # e.g., "./jetP1_K1_atm"
max_file_idx = int(sys.argv[2])                # e.g., 1000 -- max file reached
out_file_prefix:str = sys.argv[3] # e.g., "jet_conicalB_P1_K1"
order = int(sys.argv[4])            # e.g., 1

solver2D_from = lambda file_idx, dom_idx: readwritedatafiles.read_data_file(
  f"{output_prefix}{dom_idx}_{file_idx}.pkl")

def compute_disk_height(solver, seek_subatm=False) -> np.array:
  ''' Find disk height along the symmetry boundary'''

  # Extract (x, U) for elements with edge on axis of symmetry
  _bg = solver.mesh.boundary_groups["symmetry"]
  U_centerline = solver.state_coeffs[[bface.elem_ID for bface in _bg.boundary_faces], ...]
  x_tri = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs]
  x_centerline = x_tri[[bface.elem_ID for bface in _bg.boundary_faces], ...]
  # Compute dependent state for elements with edge on axis of symmetry
  rho = U_centerline[...,0:3].sum(axis=-1, keepdims=True)
  c = solver.physics.compute_variable("SoundSpeed", U_centerline)
  vel = U_centerline[...,3:5] / rho
  M = vel / c
  Mx = M[...,0:1]
  My = M[...,1:2]
  p = solver.physics.compute_variable("Pressure", U_centerline)
  T = solver.physics.compute_variable("Temperature", U_centerline)
  # Unwrap dependent states into 1D data
  ravel_x_centerline = x_centerline[...,0:1].ravel()
  ravel_y_centerline = x_centerline[...,1:2].ravel()
  ravel_Mx_centerline = Mx.ravel()
  ravel_My_centerline = My.ravel()
  ravel_p_centerline = p.ravel()
  ravel_rho_centerline = rho.ravel()
  ravel_c_centerline = c.ravel()
  ravel_T_centerline = T.ravel()
  # Filter y and quantities down to points lying on symmetry axis
  ravel_y_centerline = ravel_y_centerline[np.where(ravel_x_centerline == 0)]
  ravel_Mx_centerline = ravel_Mx_centerline[np.where(ravel_x_centerline == 0)]
  ravel_My_centerline = ravel_My_centerline[np.where(ravel_x_centerline == 0)]
  ravel_p_centerline = ravel_p_centerline[np.where(ravel_x_centerline == 0)]
  ravel_rho_centerline = ravel_rho_centerline[np.where(ravel_x_centerline == 0)]
  ravel_c_centerline = ravel_c_centerline[np.where(ravel_x_centerline == 0)]
  ravel_T_centerline = ravel_T_centerline[np.where(ravel_x_centerline == 0)]

  # Pack and returns
  N = ravel_My_centerline.size
  _out = np.empty((N,7))
  _out[:,0] = ravel_y_centerline
  _out[:,1] = ravel_Mx_centerline
  _out[:,2] = ravel_My_centerline
  _out[:,3] = ravel_p_centerline
  _out[:,4] = ravel_rho_centerline
  _out[:,5] = ravel_c_centerline
  _out[:,6] = ravel_T_centerline
  return _out

  ''' Example data processing (not run) '''
  # Naive upward search for Mach front
  i = 0
  N = ravel_My_centerline.size
  problem_encountered = False
  # Seek subatmospheric pressure
  if seek_subatm:
    while i < N:
      if ravel_p_centerline[i] <= 1e5:
        break
      i += 1
    else:
      problem_encountered = True
  # Max d(My)/dx gradient
  i_target = i + np.argmax(np.abs(np.diff(ravel_My_centerline[i:])))
  y_disk = 0.5 * (ravel_y_centerline[i_target] + ravel_y_centerline[i_target+1])
  return y_disk if not problem_encountered else 0.0

dom_idx = 1
file_idx = 0
solver_0 = solver2D_from(file_idx, dom_idx)
# Compute data for first file
dat0 = compute_disk_height(solver_0)
# Allocate space for data for all files
dat = np.empty((max_file_idx+1, *dat0.shape,))
t = np.empty((max_file_idx+1,))
# Insert data for first file
dat[0,...] = dat0
t[0] = solver_0.time

# Process all files
for file_idx in range(1, max_file_idx+1):
  solver_i = solver2D_from(file_idx, dom_idx)
  dat[file_idx,:,:] = compute_disk_height(solver_i)
  t[file_idx] = solver_i.time

np.savez_compressed(f"{out_file_prefix}.npz", dat=dat, t=t)

print(f"{perf_counter() - t0} seconds for extracting data.")
