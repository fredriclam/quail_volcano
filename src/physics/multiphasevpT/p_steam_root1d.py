import numpy as np
import scipy.interpolate
import scipy.optimize

from time import perf_counter
from typing import Callable

from inv_steam_tabulation import InverseTabulation
from pyXSteam.XSteam import RegionSelection

class SteamCalc():

  def __init__(self, itab, Liquid, method="brent"):
    # Input validation
    if method not in ["brent", "bisect-newton"]:
      print("Unknown method selected. Defaulting to brent.")
      method = "brent"
    self.c_m = Liquid["c_m"]
    self.K = Liquid["K"]
    self.rho0 = Liquid["rho0"]
    self.p0 = Liquid["p0"]
    self.itab = itab
    self.method = method
    # Magma constitutive equation (SI units)
    self.v_m_SI = lambda p : self.K / (self.rho0 * (self.K + p - self.p0))
    # Specific volume of water constrained by magma volume (SI units).
    # Not a priori consistent with v_pt(p,T); the condition v_w == v_pt is
    # solved to obtain the correct thermodynamic state.
    self.v_w_SI = lambda p, arhoW, arhoM: (1.0 - arhoM * self.v_m_SI(p)) / arhoW
    ''' Preprocess conservative state variables as below: '''
    # rho_mix = arhoM + arhoW
    # yM = arhoM / rho_mix
    # yW = arhoW / rho_mix
    # v_w = lambda p: (1.0 / rho_mix - yM * v_m(p)) / yW
    self.calls_outer = 0
    self.calls_inner = 0
    self.Tguess = self.itab.steam_table.criticalTemperatur()
    
  ''' Machinery for mapping conservative variables to (rho, T)'''

  def rhoT_brent_pT(self, p:float, Tlims:tuple, arhoW:float, arhoM:float,
      e_int:float):
    ''' Given bracketing values of T, compute rho, T using Brent root finding
    in 1D in p-T space. '''
    # Energy equation giving T (SI units)
    eq_T = lambda T: T - (e_int - arhoW * 1e3*self.itab.steam_table.u_pt(p, T)) \
      / (arhoM * self.c_m)
    if eq_T(Tlims[0]) * eq_T(Tlims[1]) > 0:
      T = np.nan
      rho = np.nan
    else:
      T = scipy.optimize.brenth(
        lambda T: eq_T(T), Tlims[0], Tlims[1],
        xtol=1e-3)
      rho = self.itab.steam_table.rho_pt(p, T)
    return rho, T

  def t_ph(self, p, h):
    ''' Wrapper for t_ph '''
    self.calls_inner += 1
    return self.itab.steam_table.t_ph(p,h)

  def hrho_brent_ph(self, p:float, hlims:tuple, arhoW:float, arhoM:float,
      e_int:float, xtol=1e-3):
    ''' Given bracketing values of h, compute h and rho using Brent root finding
    in 1D in p-h space. 
    Steam table computes t_ph faster than h_pt.
    Possible inaccuracy with h - pv == e in steam tables (IF97 Region 3). '''
    # Energy equation in var h (equation is in SI units), scaled to order of
    # temperature in Kelvin.
    eq_h = lambda h: self.itab.steam_table.t_ph(p, h) \
      - (e_int - arhoW * 1e3*self.itab.steam_table.u_ph(p, h)) \
      / (arhoM * self.c_m)
    # Energy equation with approximate energy (replacing volume from EOS with
    # the magma-constrained volume; rigorously, this should be the volume
    # computed from v_ph, but this eqn is faster and consistent at the soln).
    # eq_h = lambda h: self.t_ph(p, h) \
      # - (e_int - arhoW * 1e3*(h - 1e3*p*self.v_w_SI(1e6*p, arhoW, arhoM))) \
      # / (arhoM * self.c_m)
    if eq_h(hlims[0]) * eq_h(hlims[1]) > 0:
      rho = np.nan
      print([eq_h(h) for h in np.linspace(*hlims,10)])
      raise Exception("Root not bracketed in hrho_brent_ph.")
    else:
      h = scipy.optimize.brenth(
        lambda h: eq_h(h), hlims[0], hlims[1],
        xtol=xtol)
      rho = self.itab.steam_table.rho_ph(p, h)
    return h, rho

  def hrho_secant_ph(self, p:float, _, arhoW:float, arhoM:float,
      e_int:float, xtol=1e-3):
    ''' Secant method alternative to hrho_brent_ph. See documentation for
    hrho_brent_ph for details. '''

    raise NotImplementedError('''This method has been abandoned. Two guesses are
     needed for the algorithm start up, and it is not obvious how to obtain a
     high quality pair of initial guesses.''')
    # Cheap estimator for h
    h_estimate = lambda T: 1e-3*(e_int - arhoM * self.c_m * T) / arhoW \
        + 1e3*p*self.v_w_SI(1e6*p, arhoW, arhoM)
    T_crit = self.itab.steam_table.criticalTemperatur()
    

    eq_h = lambda h: self.t_ph(p, h) \
      - (e_int - arhoW * 1e3*(h - 1e3*p*self.v_w_SI(1e6*p, arhoW, arhoM))) \
      / (arhoM * self.c_m)

    h = scipy.optimize.root_scalar(eq_h,
      method="secant", x0=h0, x1=h1,
      xtol=xtol)
    rho = self.itab.steam_table.rho_ph(p, h)
    return h, rho

  def hrho_bisectnewton_ph(self, p:float, hlims:tuple, arhoW:float, arhoM:float,
      e_int:float, xtol=1e-3):
    # Coarse bisection tolerance
    bisect1_atol = 1e3
    '''
    # h(T) replacing v_pT with magma-constrained volume, consistent at the solution
    h_estimate = lambda T: 1e-3*(e[1,1] - arhoM[1,1] * scalc.c_m * T) / arhoW \
        + 1e3*p*scalc.v_w_SI(1e6*p, arhoW, arhoM[1,1])
    '''
    # T(h), inverse of the above
    T_estimate = lambda h: (e_int - arhoW * 1e3*(h 
      - 1e3*p*self.v_w_SI(1e6*p, arhoW, arhoM))) \
      / (arhoM * self.c_m)
    # Energy equality
    eq_h = lambda h: self.t_ph(p, h) \
      - (e_int - arhoW * 1e3*(h - 1e3*p*self.v_w_SI(1e6*p, arhoW, arhoM))) \
      / (arhoM * self.c_m)
    # Rough bisection
    h, r1 = scipy.optimize.bisect(eq_h,
      hlims[0], hlims[1], xtol=bisect1_atol, full_output=True)
    # Compute thermodynamically consistent T(h)
    T_thermo = self.itab.steam_table.t_ph(p, h)

    fcalls_misc = 1
    fcalls_bisection_part_1 = r1.function_calls
    fcalls_bisection_part_2 = 0
    fcalls_newton = 0

    while np.abs(T_estimate(h) - T_thermo) > xtol:
      # Select steam table region for computing cp, else continue bisection
      region = RegionSelection.region_ph(p, h)
      if region in {1,2}:
        # c_p = self.itab.steam_table.Cp_ph(p, h)
        # Cp_pt is faster in region 1 and 2, but estimate of T may not be in region
        c_p = self.itab.steam_table.Cp_pt(p, T_thermo)
      elif region == 3:
        c_p = self.itab.steam_table.Cp_ph(p, h)
      else:
        # Rebracket root
        h_low = h - (hlims[1]-hlims[0])/2**(r1.iterations)
        h_high = h + (hlims[1]-hlims[0])/2**(r1.iterations)
        # Perform bisection
        h, r2 = scipy.optimize.bisect(eq_h, h_low, h_high,
          xtol=xtol, full_output=True)
        fcalls_bisection_part_2 += r2.function_calls
        break
      # Take Newton step using c_p to compute slope
      dfdh = -arhoW / arhoM / (self.c_m/1e3) - 1.0/c_p
      h -= (T_estimate(h) - T_thermo) / dfdh
      fcalls_newton += 2
      T_thermo = self.itab.steam_table.t_ph(p, h)

    # Count number of functions calls  
    # f_calls = (fcalls_misc, fcalls_bisection_part_1,
    #   fcalls_bisection_part_2, fcalls_newton)

    rho = self.itab.steam_table.rho_ph(p, h)
    return h, rho

  def hrho_ph(self, p:float, hlims, arhoW:float, arhoM:float,
    e_int:float, xtol=1e-3):
    ''' Wrapper function for delegating computation of h, rho to specific
    root finding methods.'''
    if self.method == "brent":
      return self.hrho_brent_ph(p, hlims, arhoW, arhoM, e_int, xtol=xtol)
    # elif self.method == "secant":
      # return self.hrho_secant_ph(p, None, arhoW, arhoM, e_int, xtol=xtol)
    elif self.method == "bisect-newton":
      return self.hrho_bisectnewton_ph(p, hlims, arhoW, arhoM, e_int, xtol=xtol)
    else:
      raise NotImplementedError

  def compute_rho_T(self, p:float, arhoW:float, arhoM:float, e_int:float,
      is_T_needed:bool=False, xtol:float=1e-3):
    ''' Compute rho and, if needed, T. Will handle cases of pressure being above
    and below the critical pressure and will handle phase change thermodynamics
    if below critical pressure).
    
    Side effects:
    Updates memory self.Tguess if is_T_needed is True
    '''
    if p >= self.itab.steam_table.criticalPressure():
      # Use full span of temperature available in steam tables for supercrit
      # Note: temperature span for rhoT_brent_pT: (274.0, 1000).
      h, rho = self.hrho_ph(p, (100, 3600), arhoW, arhoM, e_int, xtol=xtol)
      if is_T_needed:
        T = self.t_ph(p, h)
    else:
      # Subcritical water may exist in a vapour-liquid mixture. Use specific
      # energy to differentiate phase.
      uL = self.itab.steam_table.uL_p(p)
      uV = self.itab.steam_table.uV_p(p)
      T_sat = self.itab.steam_table.tsat_p(p)
      # Compute available energy at saturation in kJ/kg
      uavail = 1e-3 * (e_int - arhoM * self.c_m * T_sat) / arhoW
      if uavail <= uL:
        # Water is less energetic than saturated liquid: condensed liquid
        # Note: temperature span for rhoT_brent_pT: (274, T_sat-0.1).
        h, rho = self.hrho_ph(p, (23, 4000), arhoW, arhoM, e_int, xtol=xtol)
        if is_T_needed:
          T = self.t_ph(p, h)
      elif uavail >= uV:
        # Water is more energetic than saturated vapour: vapour phase
        # Note: temperature span for rhoT_brent_pT: (T_sat+0.1, 1e3).
        h, rho = self.hrho_ph(p, (23, 4000), arhoW, arhoM, e_int, xtol=xtol)
        if is_T_needed:
          T = self.t_ph(p, h)
      else:
        # Water is mixture of liquid and vapour at the saturation temperature
        T = T_sat
        # Phase fraction decomposition
        yv = (uavail - uL) / (uV - uL)
        v = (1.0 - yv) * self.itab.steam_table.vL_p(p) \
          + yv * self.itab.steam_table.vV_p(p)
        rho = 1.0 / v
    if is_T_needed:
      self.Tguess = T
      return rho, T
    else:
      return rho

  def saturation_condition_p(self, p:float, arhoW:float, arhoM:float,
      e_int:float, xtol:float=1e-3):
    self.calls_outer += 1
    ''' Compute summed volume fraction minus unity at a pressure iterate p '''
    # Fix magma volume fraction (satisfy magma mass equation, magma EOS)
    alphaM = arhoM * self.v_m_SI(1e6*p)
    # Compute temperature and density (satisfy energy equation, both EOS)
    # T, rhoW = compute_T_rho(p, arhoW, arhoM, e_int, itab, c_m, v_w)
    # Compute water density (satisfy energy equation, both EOS)
    rhoW = self.compute_rho_T(p, arhoW, arhoM, e_int, is_T_needed=False,
      xtol=xtol)
    # Fix water volume fraction (satisfy water mass equation, water EOS)
    alphaW = arhoW / rhoW
    # Compute compatibility function (want == 0 to satisfy saturation)
    return alphaM + alphaW - 1.0

  def volfracCompatSolve1D(self, arhoW:float, arhoM:float, e_int:float,
    xtolouter:float=1e-3, xtolinner:float=1e-3):
    ''' Solve for thermodynamic state by 1D root-finding the compatibility (aka
    saturation) condition, i.e., that the volume fractions of water and magma sum
    to unity. '''
    # Set 1D function to solve
    f = lambda p: self.saturation_condition_p(p, arhoW, arhoM, e_int, xtol=xtolinner)
    # Set limits to pressure due to steam tables
    plim = (0.1, 100)
    if f(plim[0]) * f(plim[1]) > 0:
      return np.nan, np.nan, np.nan

    # Root-find the saturation condition for pressure (low accuracy when mixture
    # has low compressibility)
    p = scipy.optimize.brenth(f, plim[0], plim[1], xtol=xtolouter)
    # Post-process for T, rho
    rho, T = self.compute_rho_T(p, arhoW, arhoM, e_int, is_T_needed=True)
    return p, T, rho
  
  ''' Unused helper functions '''

  def T_melt(self, p:float):
    ''' Utility function for computing melting temperature, assuming type Ih ice.'''
    # Produce melting temperature curve (cheap)
    N_T_samples = 5
    t_range = np.linspace(273.157,255,N_T_samples)
    # Compute pmelt(T) using XSteam
    p_melt_range = np.array(list(map(lambda T: 
      self.itab.steam_table.pmelt_t(T, self.itab.steam_table.TYPE_ICE_Ih), t_range)))
    # Constructor interpolator for Tmelt(p)
    T_melt = scipy.interpolate.interp1d(p_melt_range, t_range, kind='cubic')
    return T_melt(p)

  def Trho_fpi(self, p:float, T_est:float, arhoW:float, arhoM:float, e_int:float):
    ''' Fixed point iteration for computing T. Not robust. Use Trho_brent instead.'''
    # Since T is relatively insensitive to pressure at fixed energy, only
    # several fixed point iterations should be needed.
    # One way to obtain an initial estimate is a linearization such as:
    #   T_est = (e_int - arhoW * (e_water_crit - cv_water_nominal * itab.T_crit))\
    #           / (arhoM * c_m + arhoW * cv_water_nominal)
    fpi_T = lambda T: (e_int - arhoW * 1e3*self.itab.steam_table.u_pt(p, T)) \
        / (arhoM * self.c_m)
    T = T_est
    # Compute fixed number of fixed point iterations
    for i in range(10):
      T_est = T
      T = fpi_T(T)
      if np.abs(T-T_est) < 1e-6:
        break
      if i >= 9:
        print("Max number of iterations reached in Trho_fpi.")
    rho = self.itab.steam_table.rho_pt(p, T)
    return T, rho

if __name__ == "__main__":
  class Params():
    ''' Skinny version of physics, containing magma properties. '''
    Gas = [
        {"R": 287., "gamma": 1.4},
        {"R": 8.314/18.02e-3, "c_p": 2.288e3}]
    Liquid = {"K": 10e9, "rho0": 2.5e3, "p0": 5e6,
        "E_m0": 0, "c_m": 3e3}
    Solubility = {"k": 5e-6, "n": 0.5}
  # Pick some test values
  arhoW = 200.0
  # Set up inverse tabulation for test case (near-exact solution from XSteam
  # in the forward direction, although note that their method is iterative in
  # the IAPWS region 3).
  itab = InverseTabulation(Params(), is_required_regular_grid=True)
  # Compute for chosen arhoW on redistributed mesh
  arhoM, e = itab.get_arhoM_e_on_rd_sampling(arhoW)
  volfracW = itab.get_mix_volfracWater_on_rd_sampling(arhoW)
  # Compute for chosen arhoW on regular mesh
  arhoM_reg, e_reg = itab.get_arhoM_e_on_reg_sampling(arhoW)
  volfracW_reg = itab.get_mix_volfracWater_on_reg_sampling(arhoW)

  # Setup steam calculator object
  scalc = SteamCalc(itab, Params.Liquid)

  # Compute values corresponding to feasible values on redistributed mesh
  p_out = np.zeros_like(arhoM)
  comp_count = 0
  t1 = perf_counter()
  for i in range(0, arhoM.shape[0], 2):
    for j in range(0, arhoM.shape[1], 2):
      # If the state is in the feasible set
      if np.abs(volfracW[i,j]-0.5) <= 0.5:
        p_out[i,j], _, _ = scalc.volfracCompatSolve1D(arhoW, arhoM[i,j], e[i,j])
        comp_count += 1
      else:
        p_out[i,j] = np.nan
  t2 = perf_counter()
  seconds_per_comp = (t2 - t1) / comp_count

  print(f"{comp_count} computations.")
  print(f"{1e3*seconds_per_comp} ms per computation.")
  print(p_out)
