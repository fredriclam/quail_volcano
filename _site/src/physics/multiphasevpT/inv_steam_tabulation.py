# Create 2D tabulation

import numpy as np
import scipy
from scipy.interpolate import griddata
from time import perf_counter
# Plot libraries
import matplotlib.pyplot as plt
import numpy.ma as ma
# External dependencies
from pyXSteam.XSteam import XSteam


class InverseTabulation():
  ''' Inverse tabulation of mixture properties that maps from conservative
  variables (arhoW, arhoM, e) to thermodynamic quantities including
  p, T, h, s. We call this an inverse tabulation since the easy direction
  is the mapping from (arhoW, p, h) to (arhoW, arhoM, e).
  
  Usage:
  Initialize with params from solver.physics, which has attribute Liquid.
  Liquid is a dict with fields "K", "rho0", "p0" that relates pressure to
  density of magma.
  Given partial density of water (arhoW), @get_arhoM_e returns tabulated values
  of arhoM and e corresponding to p, h in the instance's attribute `rd_tab`.
  `rd_tab` is a dict containing associated values of p, h, T, u, s, etc., where
  u is the specific energy of water (and not particle velocity).
  '''
  # Critical properties of water
  T_crit = 647.096 # K
  p_crit = 22.064  # MPa
  # Multiplier for relative density of sample points within the vapour dome
  dome_weight = 15.0
  # Multiplier for relative density of sample points for subcritical pressures
  p_subcrit_weight = 3.0
  # Power for concentrating points near saturated liquid curve (0 for uniform)
  hL_conc_power = 3.0

  def __init__(self, params, is_required_regular_grid:bool=False):
    '''
    Inputs:
      params, physics of the mixture, with attribute Liquid:dict for magma props
      is_required_regular_grid: flag to populate on a regular grid for plotting

    Attributes:
      steam_table:XSteam, XSteam steam table object.
      reg_tab:dict, table with regular p, h grid.
      rd_tab:dict, table associated p, h values to water phasic variables.
      is_required_regular_grid: 
    '''

    self.is_required_regular_grid = is_required_regular_grid
    # Instantiate steam table
    steam_table = XSteam(XSteam.UNIT_SYSTEM_BARE)
    self.steam_table = steam_table
    # Extract magma properties
    params_m = params.Liquid
    # Alias class attributes
    T_crit = self.__class__.T_crit
    p_crit = self.__class__.p_crit
    dome_weight = self.__class__.dome_weight
    p_subcrit_weight = self.__class__.p_subcrit_weight
    hL_conc_power = self.__class__.hL_conc_power
    self._setup_time = perf_counter()

    ''' Create regular (evenly spaced) sampling mesh. '''
    # Set regular table range and resolution based on limits of XSteam
    # h_vec = np.linspace(150, 3271, 72)
    # p_vec = np.linspace(1, 100, 33)   
    h_vec = np.linspace(150, 3271, 303)
    p_vec = np.linspace(1, 100 - 1e-10, 100)
    # h_vec = np.linspace(150, 3271, 103)
    # p_vec = np.linspace(1, 100 - 1e-10, 30)
    reg_tab = {}
    reg_tab["p"], reg_tab["h"] = np.meshgrid(p_vec, h_vec)
    
    ''' Create redistributed sampling mesh (densify points in vapour dome). '''
    rd_tab = {}
    rd_tab["p"] = reg_tab["p"].copy()
    rd_tab["h"] = reg_tab["h"].copy()
    # Redistribute p sampling to be denser below critical pressure
    if np.max(p_vec) > p_crit:
      p_subcrit_proportion = p_subcrit_weight * (p_crit - np.min(p_vec))
      p_supcrit_proportion = np.max(p_vec) - p_crit
      N_p_subcrit = np.ceil(p_subcrit_proportion / 
        (p_subcrit_proportion + p_supcrit_proportion) * p_vec.shape[0]) \
        .astype(int)
      # Uniform, dense spacing for subcritical pressures
      p_subcrit_vec = np.linspace(np.min(p_vec), p_crit, N_p_subcrit)[:-1]
      # Use smooth density variation near critical pressure
      p_supcrit_vec = np.linspace(p_crit, np.max(p_vec), len(p_vec)-N_p_subcrit+1)
      dp_subcrit = p_subcrit_vec[1] - p_subcrit_vec[0]
      dp_supcrit = p_supcrit_vec[1] - p_supcrit_vec[0]      
      ind_array = np.array(range(len(p_supcrit_vec)))
      # Set nominal spacings
      dp_supcrit_vec = dp_supcrit + (dp_subcrit - dp_supcrit) * np.power(1.0 + ind_array[:-1], -0.5)
      # Rescale
      dp_supcrit_vec *= (np.max(p_vec) - p_crit) / np.sum(dp_supcrit_vec)
      p_supcrit_vec[1:] = p_supcrit_vec[0] + np.cumsum(dp_supcrit_vec)
      p_vec_redistributed = np.hstack([p_subcrit_vec, p_supcrit_vec])
      rd_tab["p"], _ = np.meshgrid(p_vec_redistributed, h_vec)

    # Scan along lines of constant p
    for j in range(len(p_vec)):
      p_loc = rd_tab["p"][0,j]
      h_max = np.max(h_vec)
      h_min = np.min(h_vec)
      h_redistributed = h_vec.copy()
      # Vapour dome detection
      if p_loc < p_crit:
        hL = steam_table.hL_p(p_loc)
        hV = steam_table.hV_p(p_loc)
        # 3-segment distribution: constant interval lengths within each segment
        # Compute length of each of the three segments (possibly zero)
        segment_lengths = np.clip(np.array(
          [hL - h_min, (hV - hL), h_max - hV]), 0, None)
        # Compute node density weighting for each segment
        segment_weights = np.clip(np.array(
          [hL - h_min, dome_weight*(hV - hL), h_max - hV]), 0, None)
        # Compute number of intervals (of length delta h) within each segment
        intervals_per_segment = np.ceil((len(h_vec)-1) * segment_weights
          / np.sum(segment_weights)).astype(int)
        # Compute length (delta h) of each spacing (array size 3)
        interval_lengths = segment_lengths/intervals_per_segment
        # Remove extra nodes from most populated segments
        while np.sum(intervals_per_segment) > len(h_vec)-1:
          intervals_per_segment[np.argmax(intervals_per_segment)] -= 1
        # Unroll interval lengths for each interval
        dh_left = [interval_lengths[0] 
          for _ in range(intervals_per_segment[0])]
        dh_right = [interval_lengths[2] 
          for _ in range(intervals_per_segment[2])]
        # Concentrate weighting near saturated liquid curve
        ind_array = np.array(range(intervals_per_segment[1]))
        # Set density according to power of inverse-distance
        dh_dome = np.power((1.0 + ind_array)/ind_array.shape[0], hL_conc_power)
        dh_dome *= (hV - hL) / np.sum(dh_dome)
        dh = np.array([*dh_left, *dh_dome, *dh_right])
        # Construct redestributed h for p = p_loc
        h_redistributed[1:] = h_redistributed[0] + np.cumsum(dh)
      rd_tab["h"][:,j] = h_redistributed

    ''' Tabulate water phasic properties on redistributed mesh.
    Tabulate T, rho_w, u_w, s_w. Quantities T, rho_w, u_w (specific energy of
    water) define vector quantities f1, f2, where
      f1 = f1(p, T),
    and 
      f2 = f2(p, T, rho_w, u_w).
    Tabulate f1 and f2, which are used during runtime in the relation
      f1 + arhoW * f2 = [arhoM; e_int],
    where arhoW is a required input.
    '''
    # Define magma density as a function of p
    rho_m = lambda p : params_m["rho0"] * (
      1.0 + 1.0 / params_m["K"] * (p - params_m["p0"]))
    # Define f1, f2 as a function of SI thermo quantities
    f1 = lambda p, T: np.array([rho_m(p), rho_m(p) * params_m["c_m"] * T, ])
    f2 = lambda p, T, rho_w, u_w: np.array(
        [-rho_m(p)/rho_w, u_w - rho_m(p)/rho_w * params_m["c_m"] * T, ])
    # Define function to populate table
    def populate_table(tab):
      # Initialize table data for specified fields
      tab_fields = ["u_ph", "t_ph", "v_ph", "rho_ph",
        "x_ph", "s_ph", "mix_volfracWater"]
      tab_vecfields = ["f1", "f2"]
      for key in tab_fields:
        tab[key] = np.zeros((len(h_vec), len(p_vec), ))
      for key in tab_vecfields:
        tab[key] = np.zeros((2, len(h_vec), len(p_vec),))
      # Populate table
      for i in range(tab["p"].shape[0]):
        for j in range(tab["p"].shape[1]):
          # Extract non-SI p, h
          p_loc = tab["p"][i,j]
          h_loc = tab["h"][i,j]
          # Query steam table
          tab["t_ph"][i,j] = steam_table.t_ph(p_loc, h_loc)
          tab["u_ph"][i,j] = steam_table.u_ph(p_loc, h_loc)
          tab["s_ph"][i,j] = steam_table.s_ph(p_loc, h_loc)
          # Compute dependent variables
          # Note: the following computation is inaccurate and inconsistent with
          # the steam table despite the definition h = u + pv.
          # tab["v_ph"][i,j] = 1e-3 * (h_loc - tab["u_ph"][i,j]) / p_loc
          tab["v_ph"][i,j] = steam_table.v_ph(p_loc, h_loc)
          tab["rho_ph"][i,j] = 1.0 / tab["v_ph"][i,j]
          # Compute vapour mass fraction for subcritical fluid
          if p_loc < p_crit:
            tab["x_ph"][i,j] = steam_table.x_ph(p_loc, h_loc)
          else:
            tab["x_ph"][i,j] = np.nan
          # Compute precomputed vector quantities
          tab["f1"][:,i,j] = f1(1e6*p_loc, tab["t_ph"][i,j])
          tab["f2"][:,i,j] = f2(1e6*p_loc, tab["t_ph"][i,j],
            tab["rho_ph"][i,j], 1e3*tab["u_ph"][i,j])
    # Populate tables
    populate_table(rd_tab)
    if is_required_regular_grid:
      populate_table(reg_tab)
    self.rd_tab = rd_tab
    self.reg_tab = reg_tab
    # Timing
    self._setup_time = perf_counter() - self._setup_time
    # Save auxiliary variables
    self._p_vec = p_vec
    self._h_vec = h_vec

  def get_arhoM_e_on_rd_sampling(self, arhoW):
    return (*(self.rd_tab["f1"] + arhoW*self.rd_tab["f2"]),)
  
  def get_arhoM_e_on_reg_sampling(self, arhoW):
    return (*(self.reg_tab["f1"] + arhoW*self.reg_tab["f2"]),)
  
  def get_mix_volfracWater_on_rd_sampling(self, arhoW):
    return arhoW * self.rd_tab["v_ph"]
  
  def get_mix_volfracWater_on_reg_sampling(self, arhoW):
    return arhoW * self.reg_tab["v_ph"]

if __name__ == "__main__":  
  # Simulated input
  class Params():
    ''' Skinny version of physics, containing magma properties. '''
    Gas = [
        {"R": 287., "gamma": 1.4},
        {"R": 8.314/18.02e-3, "c_p": 2.288e3}]
    Liquid = {"K": 10e9, "rho0": 2.5e3, "p0": 5e6,
        "E_m0": 0, "c_m": 3e3}
    Solubility = {"k": 5e-6, "n": 0.5}
  arhoW = 200

  ''' Plot forward mapping '''
  itab = InverseTabulation(Params(), is_required_regular_grid=True)
  arhoM, e = itab.get_arhoM_e_on_rd_sampling(arhoW)
  volfracW = itab.get_mix_volfracWater_on_rd_sampling(arhoW)

  arhoM_reg, e_reg = itab.get_arhoM_e_on_reg_sampling(arhoW)
  volfracW_reg = itab.get_mix_volfracWater_on_reg_sampling(arhoW)

  # Plot forward and inverse mapping, for regular and redistributed sampling
  plt.figure(1)
  plt.subplot(2,2,1)
  plt.contourf(itab.reg_tab["h"], itab.reg_tab["p"], itab.reg_tab["t_ph"])
  plt.scatter(itab.reg_tab["h"].ravel(), itab.reg_tab["p"].ravel(), c="black", s=0.1)
  plt.xlabel("h (kJ/kg)")
  plt.ylabel("p (MPa)")
  plt.subplot(2,2,2)
  plt.contourf(arhoM_reg, e_reg, itab.reg_tab["t_ph"])
  plt.scatter(arhoM_reg.ravel(), e_reg.ravel(), c="black", s=0.1)
  plt.xlabel("partial density magma (kg/m^3)")
  plt.ylabel("mixture internal energy (J/m^3)")
  plt.xlim(left=0)
  plt.ylim(bottom=0)
  plt.subplot(2,2,3)
  plt.contourf(itab.rd_tab["h"], itab.rd_tab["p"], itab.rd_tab["t_ph"])
  plt.scatter(itab.rd_tab["h"].ravel(), itab.rd_tab["p"].ravel(), c="black", s=0.1)
  plt.xlabel("h (kJ/kg)")
  plt.ylabel("p (MPa)")
  plt.subplot(2,2,4)
  plt.contourf(arhoM, e, itab.rd_tab["t_ph"])
  plt.scatter(arhoM.ravel(), e.ravel(), c="black", s=0.1)
  plt.xlabel("partial density magma (kg/m^3)")
  plt.ylabel("mixture internal energy (J/m^3)")
  plt.xlim(left=0)
  plt.ylim(bottom=0)

  # Plot thermodynamic loci on redistributed temperature plot
  plt.figure(2)
  plt.subplot(1,2,1)
  plt.contourf(itab.rd_tab["h"], itab.rd_tab["p"], itab.rd_tab["t_ph"])
  plt.contour(itab.rd_tab["h"], itab.rd_tab["p"], arhoM > 0.0, colors='orange')
  plt.contour(itab.rd_tab["h"], itab.rd_tab["p"], arhoW * itab.rd_tab["v_ph"] < 1.0, colors='red')
  plt.contour(itab.rd_tab["h"], itab.rd_tab["p"], itab.rd_tab["t_ph"] > itab.T_crit, colors='blue')
  plt.contour(itab.rd_tab["h"], itab.rd_tab["p"], itab.rd_tab["p"] > itab.p_crit, colors='cyan')
  plt.scatter(itab.rd_tab["h"], itab.rd_tab["p"], s=0.1, c="k")

  p_below_crit = itab._p_vec[np.where(itab._p_vec < itab.p_crit)]
  hVsat_curve = np.array([itab.steam_table.hV_p(p) for p in p_below_crit])
  hLsat_curve = np.array([itab.steam_table.hL_p(p) for p in p_below_crit])
  plt.plot(hLsat_curve, p_below_crit, 'brown')
  plt.plot(hVsat_curve, p_below_crit, 'orange')

  plt.subplot(1,2,2)
  plt.contourf(arhoM, e, itab.rd_tab["t_ph"])
  plt.contour(arhoM, e, arhoM > 0.0, colors='orange')
  plt.contour(arhoM, e, arhoW * itab.rd_tab["v_ph"] < 1.0, colors='red')
  plt.contour(arhoM, e, itab.rd_tab["t_ph"] > itab.T_crit, colors='blue')
  plt.contour(arhoM, e, itab.rd_tab["p"] > itab.p_crit, colors='cyan')
  plt.scatter(arhoM, e, s=0.1, c="k")

  belowhV = np.zeros_like(itab.rd_tab["p"])
  belowhL = np.zeros_like(itab.rd_tab["p"])
  for i in range(itab.rd_tab["p"].shape[0]):
    for j in range(itab.rd_tab["p"].shape[1]):
      p = itab.rd_tab["p"][i,j]
      h = itab.rd_tab["h"][i,j]
      if p < itab.p_crit:
        belowhV[i,j] = h < itab.steam_table.hV_p(p)
        belowhL[i,j] = h < itab.steam_table.hL_p(p)
      else:
        belowhV[i,j] = np.nan
        belowhL[i,j] = np.nan
  plt.contour(arhoM, e, belowhL, colors='brown')
  plt.contour(arhoM, e, belowhV, colors='orange')
  plt.xlim(left=0)
  plt.ylim(bottom=0)

  print(f"Wall clock setup: {itab._setup_time}")

  plt.show()