''' Test script for steam calculator (p_steam_root1D.py). 
This script takes a long time to run, and must first be configured with the
path to a solver file. The script reads the solver.physics object to obtain
physics parameters. Then, a SteamCalc object is instantiated (along with its
InverseTabulation object). The SteamCalc pressure calculation, which performs
two-level 1D root finding, is performed for each point in the
InverseTabulation.
'''

import numpy as np
import matplotlib.pyplot as plt
import processing.readwritedatafiles as readwritedatafiles
from time import perf_counter
from physics.multiphasevpT.p_steam_root1d import SteamCalc
from physics.multiphasevpT.inv_steam_tabulation import InverseTabulation

# Import any solver for the physics module
solver = readwritedatafiles.read_data_file(f"v0_pboundary_0.pkl")
physics = solver.physics

# Construct steam calculator object (and inverse tabulation; may take ~ 30 sec)
sc = SteamCalc(InverseTabulation(physics, is_required_regular_grid=True),
  physics.Liquid, method="brent")

''' Test SteamCalc algorithm '''

# Set values of arhoW to test
arhoW_list = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, *np.linspace(100,1000,11)]
# Stride for downsampled grid (less total comp) --
#   expected wallclock scales with N^2 (~ 1 minute for stride=12)
stride = 1

# Set output array shape
out_shape = (len(arhoW_list),
  *sc.itab.reg_tab["p"][0:-1:stride,0:-1:stride].shape,)
# Allocate out arrays
p_out = np.nan * np.empty(out_shape)
T_out = np.nan * np.empty(out_shape)
rho_out = np.nan * np.empty(out_shape)
wallt = np.nan * np.empty(out_shape)
# Initialize progress trackers
N_total = np.prod(out_shape)
comp_count = 0
print_every = np.max([200, np.floor(N_total/20).astype(int)])
# Failure indices
fail_indices = []
exceptions = []

for k, arhoW in enumerate(arhoW_list):
  # Get forward computation values
  arhoM_reg, e_reg = sc.itab.get_arhoM_e_on_reg_sampling(arhoW)
  volfracW_reg = sc.itab.get_mix_volfracWater_on_reg_sampling(arhoW)
  # Create downsampling of forward-computed values
  p_exact_down = sc.itab.reg_tab["p"][0:-1:stride,0:-1:stride]
  arhoM_reg_down = arhoM_reg[0:-1:stride,0:-1:stride]
  e_reg_down = e_reg[0:-1:stride,0:-1:stride]

  for (i,j), _ in np.ndenumerate(p_exact_down):
    # Perform computation where forward-computed coordinates are valid
    if arhoM_reg_down[i,j] >= 0 and e_reg_down[i,j] >= 0:
      t1 = perf_counter()
      try:
        p_out[k,i,j], T_out[k,i,j], rho_out[k,i,j] = sc.volfracCompatSolve1D(
          arhoW, arhoM_reg_down[i,j], e_reg_down[i,j])
        wallt[k,i,j] = perf_counter() - t1
      except Exception as ex:
        print(f"Warn: compute failed at k,i,j={(k,i,j)}")
        fail_indices.append((k,i,j))
        exceptions.append(ex)
      # Progress tracking
      comp_count += 1
      if np.mod(comp_count, print_every) == 0:
        print(f"{comp_count / N_total * 100:.1f} %")
    else:
      N_total -= 1

''' Collate data '''

# Allocate collated data structures
p_exact_all = np.zeros_like(p_out)
arhoM_all = np.zeros_like(p_out)
e_all = np.zeros_like(p_out)
arhoW_all = np.zeros_like(p_out)
# Postprocess data
for k, arhoW in enumerate(arhoW_list):
  arhoM_reg, e_reg = sc.itab.get_arhoM_e_on_reg_sampling(arhoW)
  volfracW_reg = sc.itab.get_mix_volfracWater_on_reg_sampling(arhoW)
  # Create downsampling of forward-computed values
  p_exact_down = sc.itab.reg_tab["p"][0:-1:stride,0:-1:stride]
  arhoM_reg_down = arhoM_reg[0:-1:stride,0:-1:stride]
  e_reg_down = e_reg[0:-1:stride,0:-1:stride]
  # Save downsampling
  p_exact_all[k,...] = p_exact_down
  arhoM_all[k,...] = arhoM_reg_down
  e_all[k,...] = e_reg_down
  arhoW_all[k,...] = arhoW*np.ones_like(e_reg_down)

''' Generate plots '''

# Scatter plot wall clock
fig = plt.figure(1)
ax = plt.axes(projection='3d')
sca = ax.scatter3D(arhoM_all.ravel(), e_all.ravel(),
  arhoW_all.ravel(), '.', c=1e3*wallt[:,:,:].ravel(), s=1e3*wallt[:,:,:].ravel(), cmap="magma")
plt.colorbar(sca, label="Wall clock (ms)")
plt.xlabel("arhoM")
plt.ylabel("e")
ax.set_zlabel("arhoW (exsolved)")

# Scatter plot p
fig = plt.figure(2)
ax = plt.axes(projection='3d')
sca = ax.scatter3D(arhoM_all.ravel(), e_all.ravel(),
  arhoW_all.ravel(), '.', c=p_out.ravel(), cmap="magma")
plt.colorbar(sca, label="Pressure (Pa)")
plt.xlabel("arhoM")
plt.ylabel("e")
ax.set_zlabel("arhoW (exsolved)")

# Scatter plot T
fig = plt.figure(3)
ax = plt.axes(projection='3d')
sca = ax.scatter3D(arhoM_all.ravel(), e_all.ravel(),
  arhoW_all.ravel(), '.', c=T_out.ravel(), cmap="magma")
plt.colorbar(sca, label="Temperature (K)")
plt.xlabel("arhoM")
plt.ylabel("e")
ax.set_zlabel("arhoW (exsolved)")

# Histogram wall clock
plt.figure(4)
plt.clf()
idx_slow = np.where(wallt > 50e-3)
ax = plt.axes(projection='3d')
sca = ax.scatter3D(arhoM_all[idx_slow].ravel(), e_all[idx_slow].ravel(),
  arhoW_all[idx_slow].ravel(), '.', c=arhoW_all[idx_slow].ravel(), cmap="magma")
# plt.colorbar(sca, label="Pressure (Pa)")
plt.colorbar(sca)


''' Save outputs '''
np.save('water_test_range.npy', [p_out, T_out, rho_out, wallt], allow_pickle=True)
np.save('water_test_range_ref.npy', [p_exact_all, arhoM_all, e_all, arhoW_all], allow_pickle=True)
# To load np arrays from disk:
# res_out = np.load('water_test_range.npy', allow_pickle=True)
# res_ref = np.load('water_test_range_ref.npy', allow_pickle=True)