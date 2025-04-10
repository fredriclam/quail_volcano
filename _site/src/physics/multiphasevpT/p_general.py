import numpy as np
import scipy.interpolate
import scipy.optimize

from time import perf_counter
from typing import Callable

from physics.multiphasevpT.inv_steam_tabulation import InverseTabulation
from pyXSteam.XSteam import RegionSelection

class SteamCalc():

  def __init__(self, itab, Liquids, Gases, pStrainFree=None, method="brent"):
    # Input validation
    if method not in ["brent"]:
      print("Unknown method selected. Defaulting to brent.")
      method = "brent"
    
    # Unpack dict[str->dict[str->float]]
    self.c_c = Liquids["c"]["c_v"]
    self.K_c =   Liquids["c"]["K"]
    self.rho_c0 =   Liquids["c"]["rho0"]
    # Magma linearized equation of state (SI units)
    self.v_c_SI = lambda p : self.K_c / (self.rho_c0 * (self.K_c + p - self.p0))
    self.c_m = Liquids["m"]["c_v"]
    self.K_m = Liquids["m"]["K"]
    self.rho_m0 = Liquids["m"]["rho0"]
    # Get pressure linearization set point (common to both condensed phases)
    self.p0 =   Liquids["m"]["p0"]
    # Magma linearized equation of state (SI units)
    self.v_m_SI = lambda p : self.K_m / (self.rho_m0 * (self.K_m + p - self.p0))

    # Unpack gas assuming all entries filled in
    self.c_va = Gases[0]["c_v"]
    self.c_pa = Gases[0]["c_p"]
    self.R_a = Gases[0]["R"]
    self.gamma = Gases[0]["gamma"]
    if np.abs((self.c_pa - self.c_va)/ (self.R_a) - 1.0) > 1e-5:
      raise Exception("R != c_p - c_v within tolerance.")
    if np.abs((self.c_pa - self.c_va*self.gamma)/ (self.c_pa)) > 1e-5:
      raise Exception("gamma != c_p / c_v within tolerance.")
    self.v_a_SI = lambda p, T: self.R_a * T / p

    # Check for definition of a strain-free pressure for deformation energy
    # decomposition
    p1 = self.p0 if pStrainFree is None else pStrainFree
    self.p1 = p1

    # Save loaded properties
    self.itab = itab
    self.method = method

    self.last_h = None
    
    class DeformationEnergyFn():
      ''' Computes specific deformation energy as a function.
      Also contains useful subfunctions for decomposing the deformation en.'''
      def __init__(self, rho0, K, p0, p1):
        self.rho0 = rho0
        self.K = K
        self.p0 = p0
        self.rho1 = rho0*(1 + (p1 - p0) / K)

      def __call__(self, p):
        ''' Specific deformation energy (units m^2 / s^2). '''
        return self.strain_energy(p) + self.prestress_energy(p) - self.prestress_energy(p1)

      def strain_energy(self, p):
        ''' Strain energy as integral of (p_1-p) dv'''
        u = (p1-p)/(p+self.K-self.p0)
        return self.K/self.rho0 * (u - np.log(1 + u))

      def prestress_energy(self, p):
        ''' Prestress work as integral of -p_1 dv '''
        return p1*(1/self.rho1 - 1/(self.rho0 + self.rho0/self.K*(p-self.p0)))

      def strain_energy_quadrapprox(self, p):
          ''' Quadratic approximation of strain energy near p_1. '''
          return 0.5*self.K/self.rho0*(p-p1)**2.0 / ((p1+self.K-self.p0))**2.0

      def linearized_strain(self, p):
          return self.K*(p-p1)/ ((p1+self.K-self.p0))**2.0

    # Define specific energies (energy per mass; units of m^2/s^2)
    self.e_m = DeformationEnergyFn(self.rho_m0, self.K_m, self.p0, self.p1)
    self.e_c = DeformationEnergyFn(self.rho_c0, self.K_c, self.p0, self.p1)
    self.e_a = lambda p, T: self.c_va * T

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
    
  ''' Machinery for mapping conservative variables to (rho, T)
  For more, see p_steam_root1d.py
  '''

  def t_ph(self, p, h):
    ''' Wrapper for t_ph. '''
    self.calls_inner += 1
    return self.itab.steam_table.t_ph(p,h)
  
  def rho_ph(self, p, h):
    ''' Wrapper for rho_ph. '''
    return self.itab.steam_table.rho_ph(p, h)

  def energy_residual(self, p, h, e_water_avail):
      ''' Defines mixture energy equality residual, normalized. '''
      T = self.itab.steam_table.t_ph(p, h)
      rho_wv = self.itab.steam_table.rho_ph(p, h)
      return ((1e3*h - 1e6*p/rho_wv) - e_water_avail(p, T))/1e9

  def calculate_h(self, p:float, hlims:tuple, e_water_avail, xtol=1e-3):
    ''' Given bracketing values of h, compute h using Brent root finding
    in 1D in p-h space. 
    Note that steam table computes t_ph faster than h_pt.
    Possible inaccuracy with h - pv == e in steam tables (IF97 Region 3). '''

    res = lambda h: self.energy_residual(p, h, e_water_avail)

    if res(hlims[0]) * res(hlims[1]) > 0:
      print("Root not bracketed. Energy equation residuals: ")
      print([res(h) for h in np.linspace(*hlims,10)])
      # import matplotlib.pyplot as plt
      # plt.plot(np.linspace(*hlims,100), [energy_residual(p, h) for h in np.linspace(*hlims,100)])
      # plt.figure()
      # plt.plot(np.linspace(273,1000,100), [e_water_avail(p, T) for T in np.linspace(273,1000,100)])
      # plt.show()
      raise Exception("Root not bracketed in get_hrho_ph.")
    else:
      h = scipy.optimize.brenth(
        lambda h: res(h),
        hlims[0],
        hlims[1],
        xtol=xtol)
    return h

  def compute_rho_T(self, p:float, e_water_avail, is_T_needed:bool=False, xtol:float=1e-3):
    ''' Compute rho and, if needed, T. Will handle cases of pressure being above
    and below the critical pressure and will handle phase change thermodynamics
    if below critical pressure).
    
    Side effects:
    Updates memory self.Tguess if is_T_needed is True
    Updates self.last_h
    '''
    if p >= self.itab.steam_table.criticalPressure():
      # Use full span of temperature available in steam tables for supercrit
      # Note: temperature span for rhoT_brent_pT: (274.0, 1000).
      h = self.calculate_h(p, (100, 3600), e_water_avail, xtol=xtol)
      rho = self.rho_ph(p, h)
      if is_T_needed:
        T = self.t_ph(p, h)
    else:
      # Subcritical water may exist in a vapour-liquid mixture. Use specific
      # energy to differentiate phase.
      uL = self.itab.steam_table.uL_p(p)
      uV = self.itab.steam_table.uV_p(p)
      T_sat = self.itab.steam_table.tsat_p(p)
      # Compute available energy at saturation in kJ/kg (notation u == e)
      uavail = 1e-3*e_water_avail(p, T_sat)
      if uavail <= uL:
        # Water is less energetic than saturated liquid: condensed liquid
        # Note: temperature span for rhoT_brent_pT: (274, T_sat-0.1).
        h = self.calculate_h(p, (23, 4000), e_water_avail, xtol=xtol)
        rho = self.rho_ph(p, h)
        if is_T_needed:
          T = self.t_ph(p, h)
      elif uavail >= uV:
        # Water is more energetic than saturated vapour: vapour phase
        # Note: temperature span for rhoT_brent_pT: (T_sat+0.1, 1e3).
        h = self.calculate_h(p, (23, 4000), e_water_avail, xtol=xtol)
        rho = self.rho_ph(p, h)
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
        # Post-process: h
        h = (1.0 - yv) * self.itab.steam_table.hL_p(p) \
          + yv * self.itab.steam_table.hV_p(p)
    # Cache h for debug
    self.last_h = h
    if is_T_needed:
      self.Tguess = T
      return rho, T
    else:
      return rho

  def saturation_condition_p(self, arhoVec, p, e_water_avail, xtol:float=1e-3):
    ''' Compute summed volume fraction minus unity at a pressure iterate p.
    This is the same as summing the mass fraction weighted specific volumes,
    normalized by the mixture specific volume. '''

    self.calls_outer += 1
    
    # Compute water density (satisfying energy equation)
    rhoW, T = self.compute_rho_T(p, e_water_avail, is_T_needed=True, xtol=xtol)
    p_SI = 1e6*p
    # Compute vol fracs
    alphaA = arhoVec[0] * self.R_a * T / p_SI
    alphaW = arhoVec[1] / rhoW
    alphaM = arhoVec[2] * self.v_m_SI(p_SI)
    alphaC = arhoVec[3] * self.v_c_SI(p_SI)
    # Compute compatibility function (want == 0 to satisfy saturation)
    return alphaA + alphaW + alphaM + alphaC - 1.0

  def __call__(self, U:np.array, xtolouter:float=1e-3, xtolinner:float=1e-3):
    ''' Solve for thermodynamic state by 1D root-finding the compatibility (aka
    saturation) condition, i.e., that the volume fractions of water and magma sum
    to unity. '''

    # Single call TODO: iterate over each element, quadrature point
    U_pt = U[0,0,:]
    # Unroll mass
    arhoVec = U_pt[0:4]
    rho = np.sum(arhoVec)
    yVec = arhoVec / rho
    # TODO: momentum slice dotting for 2D
    E_kinetic = 0.5 * rho * np.dot(U_pt[4:5], U_pt[4:5])
    E_int = U_pt[5] - E_kinetic
    # Linear part of mixture heat capacity
    rho_c_v_linear = (arhoVec[0] * self.c_va 
      + arhoVec[2] * self.c_m 
      + arhoVec[3] * self.c_c)
    
    if yVec[1] < 1e-14:
      # Zero-water limit: use T as independent variable
      T = lambda p: (E_int - (
        arhoVec[2] * self.e_m(p) + arhoVec[3] * self.e_c(p))) / rho_c_v_linear
      raise NotImplementedError()
      # solve(T, self.p0)
    else:
      # Available specific energy for water
      e_water_avail = lambda p, T: (E_int - (rho_c_v_linear * T
        + arhoVec[2] * self.e_m(p) + arhoVec[3] * self.e_c(p))) / arhoVec[1]
    # Set rootfinding objective
    f = lambda p: self.saturation_condition_p(
      arhoVec, p, e_water_avail, xtol=xtolinner)
    # Set limits to pressure due to steam tables
    plim = (0.1, 100)
    if f(plim[0]) * f(plim[1]) > 0:
      # TODO: match output shape to input vector shape
      return np.nan, np.nan, np.nan, np.nan

    # Root-find the saturation condition for pressure (low accuracy when mixture
    # has low compressibility)
    p = scipy.optimize.brenth(f, plim[0], plim[1], xtol=xtolouter)
    # Post-process for T, rho
    rhow, T = self.compute_rho_T(p, e_water_avail, is_T_needed=True, xtol=xtolinner)
    return 1e6*p, T, rhow, self.last_h

  def reconstruct_E_internal(self, arhoVec, p, T, h_kJ):
    e_array = np.array([self.c_va * T,
      1e3*self.itab.steam_table.u_ph(p/1e6, h_kJ),
      self.c_m * T,
      self.c_c * T])
    return np.dot(e_array, arhoVec) + arhoVec[2] * self.e_m(p) + arhoVec[3] * self.e_c(p)
  
  def reconstruct_volfracs(self, arhoVec, p, T, h_kJ):
    v_array = np.array([(self.R_a * T)/ p,
      1/self.itab.steam_table.rho_ph(p/1e6, h_kJ),
      self.v_m_SI(p),
      self.v_c_SI(p)])
    return v_array * arhoVec

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
    Gases = [
        {"R": 287., "gamma": 1.4},
        {"R": 8.314/18.02e-3, "c_p": 2.288e3}]
    Gases[0]["c_v"] = Gases[0]["R"] / (Gases[0]["gamma"] - 1)
    Gases[0]["c_p"] = Gases[0]["R"] / (Gases[0]["gamma"] - 1) * Gases[0]["gamma"]
    Gases[1]["c_v"] = Gases[1]["c_p"] - Gases[1]["R"]
    Gases[1]["gamma"] = 1 / (1 - Gases[1]["R"] / Gases[1]["c_p"])
    Liquids = {
      "m": {"K": 10e9, "rho0": 2.5e3, "p0": 5e6,
        "c_m": 3e3},
      "c": {"K": 10e9, "rho0": 2.5e3, "p0": None,
        "c_m": 3e3},
    }
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
  scalc = SteamCalc(itab, Params.Liquids, Params.Gases)
  U_test = np.array([4,32,1600,200,50.0,1e9])
  U_test = np.expand_dims(U_test, axis=(0,1))
  scalc(U_test)

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
