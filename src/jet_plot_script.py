import os
import numpy as np
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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

_this_script_name = sys.argv[0]
output_prefix = sys.argv[1]         # e.g., "./jetP1_conical3_test1_atm"
i = int(sys.argv[2])                # e.g., 1000
out_figure_prefix:str = sys.argv[3] # e.g., "jet_conicalB_P1_K1"
order = int(sys.argv[4])            # e.g., 1
# Optional arg
manual_dpi = None
if len(sys.argv) > 5:
  manual_dpi = int(sys.argv[5])

# Set to True if the inner 2 domains have WriteInterval half as long
is_inner_domain_dense_output:bool = False
# True scale limit
R1 = 500
# Plot gamma exponent (controls squeezing of far-away elements)
pgamma = 0.7
# Global font size (pt)
global_fontsize = 14

solver2D_from = lambda file_idx, dom_idx: readwritedatafiles.read_data_file(
  f"{output_prefix}{dom_idx}_{file_idx}.pkl")

# Shifted diverging cmap for Mach
shift_div_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    f'trunc(bwr,{0.25},{1.0})',
    matplotlib.cm.bwr(np.linspace(0.25, 1, 100)))

# Initial solver for physics
solver = solver2D_from(0,1)
physics = solver.physics

plot_instruction_p = {
  "quantity": lambda U: physics.compute_variable(
    "Pressure", U) / 1e5,
  "clims": (0, 1.2),
  "colorbar_label": "$p$ (bar)",
  "cmap": matplotlib.cm.viridis,
  "out_figure_name": f"{out_figure_prefix}_p",
}
plot_instruction_T = {
  "quantity": lambda U: physics.compute_variable(
    "Temperature", U),
  "clims": (200, 1050),
  "colorbar_label": "$T$ (K)",
  "cmap": matplotlib.cm.inferno,
  "out_figure_name": f"{out_figure_prefix}_T",
}
plot_instruction_M = {
  "quantity": lambda U: (np.linalg.norm(U[...,3:5], axis=-1, keepdims=True)
                          / (U[...,0:3].sum(axis=-1, keepdims=True) * physics.compute_variable(
                          "SoundSpeed", U))),
  "clims": (0, 3),
  "colorbar_label": "$|M|$",
  "cmap": shift_div_cmap,
  "out_figure_name": f"{out_figure_prefix}_M",
}
plot_instruction_yM = {
  "quantity": lambda U: (U[...,2:3] / U[...,0:3].sum(axis=-1, keepdims=True)),
  "clims": (0, 1),
  "colorbar_label": r"$y_\mathrm{m}$",
  "cmap": matplotlib.cm.Oranges,
  "out_figure_name": f"{out_figure_prefix}_yM",
}
plot_instruction_c = {
  "quantity": lambda U: physics.compute_variable(
    "SoundSpeed", U),
  "clims": (50, 350),
  "colorbar_label": "Sound speed (m/s)",
  "cmap": matplotlib.cm.plasma,
  "out_figure_name": f"{out_figure_prefix}_c",
}
plot_instruction_Gamma = {
  "quantity": lambda U: physics.compute_variable(
    "Gamma", U),
  "clims": (1.0, 1.4),
  "colorbar_label": "$\Gamma$",
  "cmap": matplotlib.cm.cool_r,
  "out_figure_name": f"{out_figure_prefix}_Gamma",
}

plot_instructions = [
  plot_instruction_p,
  plot_instruction_T,
  plot_instruction_M,
  plot_instruction_yM,
  plot_instruction_c,
  plot_instruction_Gamma
]

for plot_index, plot_instruction in enumerate(plot_instructions):

  # Unpack plot instructions
  quantity = plot_instruction["quantity"]
  clims = plot_instruction["clims"]
  colorbar_label = plot_instruction["colorbar_label"]
  cmap = plot_instruction["cmap"]
  out_figure_name = plot_instruction["out_figure_name"]
  
  # Set figure
  fig, ax = plt.subplots(figsize=(9,11), dpi=100)

  # Compute for innermost domain
  solver = solver2D_from(i, 1)
  
  x_tri = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs]
  # Plot detailed
  (_, cb) = processing.mdtools.plot_detailed(x_tri, solver.state_coeffs, quantity, clims, order,
    colorbar_label=colorbar_label,
    xview=(0,1000), yview=(-300,850), n_samples=12, cmap=cmap)

  # Extract solver and plot for outermost domain
  for outer_index in [2,3,4,5,6,7,8,9]:
    if outer_index == 2 or not is_inner_domain_dense_output:
      solver = solver2D_from(i, outer_index)
    else:
      # Larger file-save interval
      solver = solver2D_from(i//2, outer_index)

    
    x_tri_2 = solver.mesh.node_coords[solver.mesh.elem_to_node_IDs]
    r_tri_2 = np.sqrt((x_tri_2**2).sum(axis=-1, keepdims=True))
    # Apply nonlinear spatial scaling outside
    x_tri_2_scaled = np.where(r_tri_2 > R1,
                              x_tri_2 * (R1 + np.clip(r_tri_2 - R1, 0, None)**pgamma) / r_tri_2,
                              x_tri_2)
    # Plot detailed
    processing.mdtools.plot_detailed(x_tri_2_scaled, solver.state_coeffs, quantity, clims, order,
      colorbar_label="",
      xview=(0,10000), yview=(-10000,10000), n_samples=4,
      add_colorbar=False, cmap=cmap)

  # Add figure details
  plt.title(f"$t$ = {solver.time:.2f} s", )
  plt.xlim(left=0)

  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(global_fontsize)
  cb.ax.tick_params(labelsize=global_fontsize)
  cb.ax.yaxis.label.set_fontsize(global_fontsize)

  # Plot separation circle
  _i = x_tri[:,:,1].ravel().argmin()
  end_angle_rad = np.arctan2(x_tri[:,:,1].ravel()[_i], x_tri[:,:,0].ravel()[_i])
  theta_range = np.linspace(np.pi/2, end_angle_rad, 250)
  circle_x_range = R1 * np.cos(theta_range)
  circle_y_range = R1 * np.sin(theta_range)
  plt.plot(circle_x_range, circle_y_range, linewidth=1, color='w')

  # Contraction (fisheye) mapping
  lin_to_pow = lambda r: np.where(r > R1, R1 + np.clip(r - R1, 0, None)**pgamma, r)
  pow_to_lin = lambda r: np.where(r > R1, R1 + np.clip(r - R1, 0, None)**(1/pgamma), r)

  # Replace y-axis with nonlinearly scaled
  sec_yaxis = ax.secondary_yaxis('left', functions=(pow_to_lin, lin_to_pow))
  sec_yaxis.yaxis.set_tick_params(labelsize=global_fontsize)
  sec_yaxis.set_yticks([*np.arange(-200,500+1,100), *np.arange(1000,9000+1,1000)])
  sec_yaxis.yaxis.set_minor_locator(AutoMinorLocator(2))
  # ax.yaxis.set_visible(False)
  plt.ylabel("$z$ (m)")
  ax.yaxis.set_ticks([100])
  # sec_yaxis.set_label("$z~(m)$")

  # Replace x-axis with nonlinearly scaled
  sec_xaxis = ax.secondary_xaxis('bottom', functions=(pow_to_lin, lin_to_pow))
  sec_xaxis.xaxis.set_tick_params(labelsize=global_fontsize)
  sec_xaxis.set_xticks([*np.arange(0,500+1,100), 3000, 6000, 9000])
  sec_xaxis.xaxis.set_minor_locator(AutoMinorLocator(2))
  # ax.xaxis.set_visible(False)
  plt.xlabel("$r$ (m)")
  ax.xaxis.set_ticks([0])
  # sec_xaxis.set_label("$r~(m)$")

  plt.draw()

  # Est. 7 minutes for 400 dpi + 800 dpi
  if manual_dpi is None:
    plt.savefig(f"{out_figure_name}_400dpi_{i}.png", format="png", dpi=400)
    plt.savefig(f"{out_figure_name}_800dpi_{i}.png", format="png", dpi=800)
  else:
    plt.savefig(f"{out_figure_name}_{manual_dpi}dpi_{i}.png", format="png", dpi=manual_dpi)
