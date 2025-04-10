'''
Dependency-free material properties .
Implements ThermodynamicMaterial, an interface that exposes heat capacities
c_v and c_p, and (p, T) dependent functions e_pT, h_pT, v_pT as well as
their partial derivatives dv_dp_isoT_pT and dv_dT_isop_pT. Two classes
of materials are implemented: ideal gases and linearized-density magma.
The deformation energy (methods relating to integral of - p dV for the
linearized-density substance).
'''

import abc
import numpy as np


# TODO: in import: mixture sound speed
# TODO: exsolution energy / energy of vaporization etc.


class ThermodynamicMaterial(metaclass=abc.ABCMeta):
  ''' Abstract class for materials with thermodynamic properties. '''
  @property
  @abc.abstractmethod
  def c_v(self):
    pass
  
  @c_v.setter
  def c_v(self, val):
    self._c_v = val

  @property
  @abc.abstractmethod
  def c_p(self):
    pass

  @c_p.setter
  def c_p(self, val):
    self._c_p = val

  @abc.abstractmethod
  def e_pT(self, p, T):
    pass

  @abc.abstractmethod
  def h_pT(self, p, T):
    pass

  @abc.abstractmethod
  def v_pT(self, p, T):
    pass

  @abc.abstractmethod
  def dv_dp_isoT_pT(self, p, T):
    ''' Partial derivative of specific volume v_w w.r.t pressure at constant T. Used in chain rule.'''
    pass

  @abc.abstractmethod
  def dv_dT_isop_pT(self, p, T):
    ''' Partial derivative of specific volume v_w w.r.t T at constant p. Used in chain rule.'''
    pass

  def rho_pT(self, p, T):
    return 1.0 / self.v_pT


class IdealGas(ThermodynamicMaterial):
  def __init__(self, c_v=1826.6237513873475, c_p=2288.0):
    self.c_v, self.c_p = c_v, c_p
    self.R, self.gamma = self.c_p - self.c_v, self.c_p / self.c_v
  
  @property
  def c_v(self):
    return self._c_v

  @c_v.setter
  def c_v(self, val):
    self._c_v = val
  
  @property
  def c_p(self):
    return self._c_p

  @c_p.setter
  def c_p(self, val):
    self._c_p = val
  
  def e_pT(self, p ,T):
    return self.c_v * T

  def h_pT(self, p, T):
    return self.c_p * T
  
  def v_pT(self, p, T):
    return self.R * T / p
  
  def dv_dp_isoT_pT(self, p, T):
    return -self.R*T/p**2

  def dv_dT_isop_pT(self, p, T):
    return self.R/p

  def dp_drho_isos_pT(self, p, T):
    return self.gamma * self.R * T
  
  def e_sv(self, s, v):
    ''' Internal energy as a thermodynamic potential, w.r.t.
    unit T0, v0, s0. Scale and shift are arbitrary since only
    derivatives of the potential are needed, and s0 can be defined
    with an arbitrary ground state. '''
    T0, v0, s0 = 1.0, 1.0, 0.0
    return self.c_v * T0 * (self.v / v0) ** (1 - self.gamma) \
      * np.exp((s - s0) / self.c_v)


class IdealGasMixture(IdealGas):
  ''' Mixing rule for ideal gases.
  Inputs:
    gases: tuple of IdealGas objects
    y: list or array of mass fractions
  Implements the partial volume partition function self.v_partition.
  '''

  def __init__(self, gases:tuple, y):
    n = len(gases)
    self.gases = gases
    # Mass fraction weighted properties
    self.R = np.dot(y, [gas.R for gas in self.gases])
    self.c_v = np.dot(y, [gas.c_v for gas in self.gases])
    self.c_p = self.c_v + self.R
    self.gamma = self.c_p / self.c_v
    self.R_frac = y * np.array([gas.R for gas in self.gases]) / self.R

  def v_partition(self):
    return self.R_frac


class DeformationEnergyFn():
  ''' Computes specific deformation energy as a function for a linearized-density
  material with zero thermal expansion.
  Also contains useful subfunctions for decomposing the deformation en.'''
  def __init__(self, rho0, K, p0, p1):
    self.rho0 = rho0
    self.K = K
    self.p0 = p0
    self.p1 = p1
    self.rho1 = rho0*(1 + (p1 - p0) / K)

  def __call__(self, p):
    ''' Specific deformation energy (units m^2 / s^2). '''
    return self.strain_energy(p) + self.prestress_energy(p) - self.prestress_energy(self.p1)

  def rho(self, p):
    return self.rho0*(1 + (p - self.p0) / self.K)

  def p(self, v):
    return self.p0 + self.K / self.rho0 *(1/v - self.rho0)

  def drhodp(self):
    ''' drho/ dp where rho is density. '''
    return self.rho0 / self.K

  def dvdp(self, p):
    ''' dv / dp where v is specific volume. '''
    return -1.0 / self.rho(p)**2 * self.drhodp()

  def strain_energy(self, p):
    ''' Strain energy as integral of (p_1-p) dv'''
    u = (self.p1-p)/(p+self.K-self.p0)
    return self.K/self.rho0 * (u - np.log(1 + u))

  def prestress_energy(self, p):
    ''' Prestress work as integral of -p_1 dv '''
    return self.p1*(1/self.rho1 - 1/(self.rho0 + self.rho0/self.K*(p-self.p0)))

  def strain_energy_quadrapprox(self, p):
      ''' Quadratic approximation of strain energy near p_1. '''
      return 0.5*self.K/self.rho0*(p-self.p1)**2.0 / ((self.p1+self.K-self.p0))**2.0

  def linearized_strain(self, p):
      return self.K*(p-self.p1)/ ((self.p1+self.K-self.p0))**2.0

  def ddp(self, p):
    ''' Derivative of total deformation energy with respect to pressure.
    Units of specific volume, (specific energy) / (volumetric energy).
    Conjugate to pressure. This can be shown using
    d/dp int(p dv) = d/dp int p dv/dp dp = p * dv/dp.
    ''' 
    return p / self.rho(p)**2 * (self.rho0 / self.K)


class MagmaLinearizedDensity(ThermodynamicMaterial):
  def __init__(self, c_v:float=3000.0, rho0:float=2.7e3, K:float=10e9,
      p_ref:float=5e6, neglect_edfm:bool=False):
    '''
    Inputs:
    neglect_edfm (optional): whether to neglect deformation energy in
      calculations for e_pT and h_pT calculations. Neglecting deformation
      energy removes pressure from the energy dependence. The internal e_dfm
      object is still needed to compute deformation properties (EOS). '''
    self.c_v, self.rho0, self.K, self.p_ref = c_v, rho0, K, p_ref
    self.neglect_edfm = neglect_edfm
    self.e_dfm = DeformationEnergyFn(rho0, K, p_ref, p_ref)
  
  @property
  def c_v(self):
    return self._c_v
  
  @c_v.setter
  def c_v(self, val):
    self._c_v = val
  
  @property
  def c_p(self):
    ''' c_p == (dh/dT)_p. 
    Since energy decomposes additively to dependence on p and T, and
    pv is independent of T, c_p = c_v. '''
    return self._c_v
  
  @c_p.setter
  def c_p(self, val):
    self._c_p = val
  
  def e_pT(self, p ,T):
    if self.neglect_edfm:
      return self.c_v * T
    return self.c_v * T + self.e_dfm(p)

  def h_pT(self, p, T):
    ''' Compute enthalpy from (dh/dT)_p and (dh/dp)_T = v'''
    return self.e_pT(p,T) + p * self.v_pT(p, T)
  
  def v_pT(self, p, T):
    return 1.0 / self.e_dfm.rho(p)
  
  def dv_dp_isoT_pT(self, p, T):
    return self.e_dfm.dvdp(p)

  def dv_dT_isop_pT(self, p, T):
    return self.e_dfm.dvdp(p)

  def dp_drho_isos_pT(self, p, T):
    return self.K / self.rho0
  
  def e_sv(self, s, v):
    ''' Internal energy as a thermodynamic potential, w.r.t.
    unit T0, v0, s0. Scale and shift are arbitrary since only
    derivatives of the potential are needed, and s0 can be defined
    with an arbitrary ground state. Compare the expression
      self.c_v * T0 * np.exp((s - s0) / self.c_v) + self.e_dfm(self.e_dfm.p(v))
    to ideal gas
      self.c_v * T0 * (self.v / v0) ** (1 - self.gamma)
        * np.exp((s- s0) / self.c_v) + self.e_dfm(self.e_dfm.p(v))
    which differ by compression coupling to the first term, and the addition
    of intermolecular energy in the second term.
    '''
    T0, s0 = 1.0, 0.0
    return self.c_v * T0 * np.exp((s - s0) / self.c_v) \
      + self.e_dfm(self.e_dfm.p(v))


class RedlichKwongConstHeat(ThermodynamicMaterial):
  ''' Empirical Redlich-Kwong equation of state with parametrization by Halbach
  & Chatterjee 82. A constant heat capacity is assumed--this is not
  thermodynamically reversible. 
  
  Two-parameter Redlich-Kwong:
            RT         a
   p =   ------ - ------------
          V - b               _
                  V(V + b) \/T
  where a = a(T), b = b(p).
  '''
  def __init__(self, cache_tol=1e-14):
    ''' Set water properties and set cache tolerance. Cache tolerance is used
    to check if volume needs to be recomputed for given input values to
    methods. '''
    # Set reference e, p, T triple for energy from fit range
    self.p0_MPa = 215 # MPa
    self.T0 = 750 # K
    # Define molar universal gas constant (J / mol K)
    self.R_mol = 8.3143
    # Molar mass
    self.mm = 18.02e-3 # kg/mol
    # Critical properties of water for nondimensionalization
    self.rho_c = 322           # kg/m^3
    self.V_c = 1e6/(self.rho_c/self.mm)  # cm^3/mol
    self.T_c = 647.096
    # Set heat capacity
    self.c_p = 3880          # Average value of Degruyter & Huber
    self.c_v = self.c_p - self.R_mol/self.mm
    self.c_v_mol, self.c_p_mol = self.c_v * self.mm, self.c_p * self.mm
    self.R, self.gamma = self.R_mol/self.mm, self.c_p / self.c_v
    # Initialize cache for molar volume
    self.cache_tol = cache_tol
    self._cache_pT = (None, None)
    self._cache_vals = None
    # Top cache values that would be used: specific volume, compressibility at (p,T)
    self._v = None
    self._Z = None
  
  @property
  def c_v(self):
    return self._c_v
  
  @property
  def c_p(self):
    return self._c_p
  
  def e_pT(self, p ,T):
    self._check_and_precompute(p, T)
    return self._cache_vals["specific"]["e"]

  def h_pT(self, p, T):
    self._check_and_precompute(p, T)
    return self._cache_vals["specific"]["h"]
  
  def v_pT(self, p, T):
    self._check_and_precompute(p, T)
    return self._v

  def dv_dp_isoT_pT(self, p, T):
    raise NotImplementedError

  def dv_dT_isop_pT(self, p, T):
    raise NotImplementedError

  def dp_drho_isos_pT(self, p, T):
    raise NotImplementedError
  
  def _check_and_precompute(self, p, T) -> None:
    ''' Validates cache for given p, T. If the cached value is stale,
    precomputes molar volume and energies at p, T.'''
    
    if np.any([val is None for val in self._cache_pT]) or \
      (p - self._cache_pT[0])**2 + (T - self._cache_pT[1])**2 > self.cache_tol**2:
      # Cache is stale; update is needed.
      self._cache_pT = (p, T)
    else:
      # Use cached values
      return

    ''' Compute values '''
    # Unpack
    R = self.R_mol
    V_c = self.V_c
    # Pa -> MPa -> bar
    p = p * 1e-6 * 1e1
    # Precompute power terms
    powers_T = [1, T, 1/T]
    powers_dT = [0, 1, -1/(T*T)]
    powers_p = [1, p, p*p]
    # Compute coefficients
    coeffs_a = [1.616e8, -4.989e4, -7.358e9]
    coeffs_b_num = [3.4505e-4, 3.8980e-9, -2.7756e-15]
    coeffs_b_den = [6.3944e-2, 2.3776e-5, 4.5717e-10]
    a = np.dot(coeffs_a, powers_T)
    dadT = np.dot(coeffs_a, powers_dT)
    b = (1 + p*np.dot(coeffs_b_num, powers_p)) / np.dot(coeffs_b_den, powers_p)
    # Define compressibility form
    Z_fn = lambda V: V/(V-b) - 1e-1 * a / (R*T*(V+b)*np.sqrt(T))
    # Cubic form for molar density (cm^3/mol), w.r.t pressure in J/cc, T in K
    poly = [1e-1*p, -R*T, -(b*b*(1e-1*p) + R*T*b - (1e-1*a)/np.sqrt(T)),
      -(1e-1*a)*b/np.sqrt(T)]
    # Dimensional vector (poly*dimensonals -> Pa cm^9 / mol^3 == J cc^2 / mol^3)
    dimensionals = [V_c**3, V_c**2, V_c, 1]
    # Define polynomial coefficients to nondimensional cubic for molar volume (V/V_c)
    poly_scaled = np.asarray(poly) * np.asarray(dimensionals)
    poly_scaled /= poly_scaled[0]

    # Solve cubic equation for nondimensional molar volume
    V_roots = np.roots(poly_scaled)
    # Filter roots: real root, V >= b
    V = np.array([np.real(root) for root in V_roots
      if np.isreal(root) and root*V_c >= b])
    # Filter roots: take gas-like for subcritical conditions
    is_using_sub_crit_filter = False
    if len(V) > 1:
      is_using_sub_crit_filter = True
      V = V[np.argmin(np.abs(Z_fn(V*V_c)-1))]
    # Cast possible array to float (np.array, float -> float, raise for list)
    V = float(V)

    # Reconstruct pressure for verification
    # p_test = [(R*T/(root*V_c - b) - 1e-1*a / (root*V_c*(root*V_c+b)*np.sqrt(T))).astype(str) + " MPa" for root in V]
    p_test = (R*T/(V*V_c - b) - 1e-1*a / (V*V_c*(V*V_c+b)*np.sqrt(T))).astype(str) + " MPa"
    is_using_sub_crit_filter,

    # Energy computation
    # p_c = 22.0641e1 # bar, Used in fit
    # # Define fitting c_v (quadratic with mixed terms and reciprocal term)
    # cv_spec_fit = lambda p, T, a0, a1, a2, an, b1, b2, bn, c1: a0 \
    # + a1*T/T_c + a2*(T/T_c)**2 + an*(T_c/T) + c1*(T/T_c) * (p/p_c) \
    # + b1*p/p_c + b2*(p/p_c)**2 + bn*(p_c/p)
    # # Integrate c_v dT
    # cv_integral = lambda p, T, a0, a1, a2, an, b1, b2, bn, c1: \
    #   T*(a0 + b1*p/p_c + b2*(p/p_c)**2 + bn*(p_c/p) 
    #   + (a1/2 + c1/2*(p/p_c))*T/T_c + (a2/3)*(T/T_c)**2 + an*(T_c/T) * np.log(T))
    # # Prefit curve coefficients, units kJ / (kg K)
    # cv_curve_coeffs = np.array([ 5.189855352275467e+00, -2.995975942745098e+00,
    #       7.112851012456277e-01,  4.036027653470066e-02,
    #      -4.560737798210990e-02,  9.108877318667112e-05,
    #      -2.024779009243538e-01,  3.802791999428569e-02])

    # if len(cv_curve_coeffs) == 0:
    #   ''' Source code for fitting over a sampled range of c_v over mg_T, mg_p values.''' 
    #   # Get curve coefficients from regression
    #   cv_curve_coeffs = scipy.optimize.curve_fit(
    #     lambda pT, a0, a1, a2, an, b1, b2, bn, c1: \
    #     cv_fit(pT[0], pT[1], a0, a1, a2, an, b1, b2, bn, c1),
    #     xdata=np.vstack((mg_p.ravel(), mg_T.ravel())),
    #     ydata=cv.ravel())[0]
    #   raise Exception("temporary")

    # T-dependent fit
    # cv_spec_fit = lambda p, T, a0, a1, a2, an: a0 + a1*T/T_c + a2*(T/T_c)**2 + an*(T_c/T)
    # cv_integral = lambda p, T, a0, a1, a2, an: T*( \
    #   a0 + a1/2*T/T_c + (a2/3)*(T/T_c)**2 + an*(T_c/T) * np.log(T))
    # # Prefit curve coefficients for T-dependent fit, units kJ / (kg K)
    # cv_curve_coeffs = np.array([4.722349908528655, -2.625419021268893,
    #   0.711285571010602, 0.040359794430233])
    # if len(cv_curve_coeffs) == 0:
    #   # Get curve coefficients from regression
    #   cv_curve_coeffs = scipy.optimize.curve_fit(cv_spec_fit, xdata=mg_T.ravel(), ydata=cv.ravel())[0]
    #   # Test c_v fit
    #   # plt.plot(mg_T.ravel(), cv.ravel(), '.')
    #   # mg_T_lin_sample = np.linspace(mg_T.min(), mg_T.max(), 100)
    #   # plt.plot(mg_T_lin_sample, cv_optfit(mg_T_lin_sample), '-')

    # Define optimal c_v curve for molar heat capacity J / (mol K)
    # cv_optfit = lambda p, T: mm * 1e3 * cv_spec_fit(p, T)
    # Convert integral of c_v dT to molar energy J / (mol K)
    # e_ideal_fn = lambda p, T: mm * 1e3 * cv_integral(p, T, *cv_curve_coeffs)
    # Ideal energy at constant heat capacity
    e_ideal_fn = lambda p,T: self.c_v_mol * T 

    ''' Compute departure functions. '''
    # Molar departure quantities divided by RT
    #   (conversion bar * cc / mol == 1e-1 J / mol)
    #   (b has units: (cc/mol)^2 * T^(1/2) * bar)
    #   (b has units of V*V_c: cc/mol)
    Z = 1e-1 * p*(V*V_c) / (R*T)
    self._Z = Z
    e_dep_nondim = 1e-1 * (1.5*a - dadT*T)/ (b*R*T*np.sqrt(T)) * np.log((V*V_c+b)/(V*V_c))
    h_dep_nondim = e_dep_nondim + 1 - Z

    if False:
      c_p = cv_optfit(p, T) + R
      e_ideal = e_ideal_fn(p, T)
      e_total = e_ideal - R*T*e_dep_nondim
      h_ideal = e_ideal + R*T
      h_total = h_ideal - R*T*h_dep_nondim
      # Apply constant shift to match IAPWS energy
      # e_total -= mm * 1e3 * (cv_integral(T_ref, *cv_curve_coeffs) - e_ref_kJkg)
      # h_total -= mm * 1e3 * (cv_integral(T_ref, *cv_curve_coeffs) - e_ref_kJkg)
      e_total += 0*273571.0359366494*mm # Fit for J/kg
      h_total += 0*273571.0359366494*mm
    else:
      # Constant heat capacity model
      c_p = 3880*self.mm # 3880 J / kgK -> J / mol K
      h_ideal = self.c_p_mol * T
      e_ideal = (self.c_p_mol - R) * T
      h_total = h_ideal - R*T*h_dep_nondim
      e_total = e_ideal - R*T*e_dep_nondim

    # Compute mass-specific volume (m^3/kg)
    v = 1e-6 * V * V_c / self.mm

    # Cache top values
    self._v = V
    self._Z = Z
    # Cache debug values
    self._cache_vals = {
      "molarSI": {
        "h_ideal": h_ideal,
        "e_ideal": e_ideal,
        "h": h_total,
        "e": e_total,
        "v": 1e-6*V*V_c,
      },
      "specific": {
        "h_ideal": h_ideal / self.mm,
        "e_ideal": e_ideal / self.mm,
        "h": h_total / self.mm,
        "e": e_total / self.mm,
        "v": v,
      },
      "Z": Z,
      "debug": {
        "Z_check": Z_fn(V*V_c),
        "a": a,
        "b": b,
        "V_c": V_c,
        "rho_c": self.rho_c,
        "c_p_molar": c_p,
        "V": V,
        "poly_scaled": poly_scaled,
        "V_roots": V_roots,
        "p_test": p_test,
        "is_using_sub_crit_filter": is_using_sub_crit_filter,
        # "fns": {
        #   "cv_optfit": cv_optfit,
        #   "e_ideal_fn": e_ideal_fn,
        # },
      }
    }
  

class EmpiricalRedlichKwongConstB(ThermodynamicMaterial):
  ''' Empirical Redlich-Kwong equation of state with parametrization
  for geological use by Holloway (1977) used together with a constant
  ideal-gas-part of heat capacity.
  
  Two-parameter Redlich-Kwong:
            RT         a
   p =   ------ - ------------
          V - b               _
                  V(V + b) \/T
  where a = a(T), and b is constant.
  '''
  def __init__(self, cache_tol=1e-14):
    ''' Set water properties and set cache tolerance. Cache tolerance is used
    to check if volume needs to be recomputed for given input values to
    methods. '''
    # Set reference e, p, T triple for energy from fit range
    self.p0_MPa = 525 # MPa
    self.T0 = 510 + 273.15 # K
    # Define molar universal gas constant (J / mol K)
    self.R_mol = 8.3143
    # Molar mass
    self.mm = 18.02e-3 # kg/mol
    # Critical properties of water for nondimensionalization
    self.rho_c = 322           # kg/m^3
    self.V_c = 1e6/(self.rho_c/self.mm)  # cm^3/mol
    self.T_c = 647.096
    # Set ideal gas part of heat capacity (function of T only)
    self.c_p = 3880          # Average value of Degruyter & Huber
    self.c_v = self.c_p - self.R_mol/self.mm
    self.c_v_mol, self.c_p_mol = self.c_v * self.mm, self.c_p * self.mm
    self.R, self.gamma = self.R_mol/self.mm, self.c_p / self.c_v
    # Initialize cache for molar volume
    self.cache_tol = cache_tol
    self._cache_pT = (None, None)
    self._cache_vals = None
    # Top cache values that would be used: specific volume, compressibility at (p,T)
    self._v = None
    self._Z = None
  
  @property
  def c_v(self, T, v):
    # Precompute power terms
    powers_T_Celsius = [1, (T-273.15), (T-273.15)**2, (T-273.15)**3]
    # Compute coefficients
    coeffs_a_Celsius = [166.8e6, -193080, 186.4, -0.071288]
    coeffs_dadT_Celsius = [-193080, 2*186.4, 3*-0.071288, 0]
    coeffs_ddadT2_Celsius = [2*186.4, 6*-0.071288, 0, 0]
    a = np.dot(powers_T_Celsius, coeffs_a_Celsius)
    da = np.dot(powers_T_Celsius, coeffs_dadT_Celsius)
    dda = np.dot(powers_T_Celsius, coeffs_ddadT2_Celsius)
    macro_coeffs = np.array([3/4, -T, T*T])/T**1.5
    # Compute molar volume
    Vmol = 1e6 * v * self.mm # cm^3/mol
    b = 14.6 # cm^3/mol
    return self._c_v - np.dot(np.array(macro_coeffs, [a, da, dda])) * np.log(Vmol/(Vmol+b))
  
  @property
  def c_p(self):
    raise NotImplementedError("Implement c_p in terms of c_v, Z.")
    return self._c_p
  
  def e_pT(self, p ,T):
    self._check_and_precompute(p, T)
    return self._cache_vals["specific"]["e"]

  def h_pT(self, p, T):
    self._check_and_precompute(p, T)
    return self._cache_vals["specific"]["h"]
  
  def v_pT(self, p, T):
    self._check_and_precompute(p, T)
    return self._v

  def dv_dp_isoT_pT(self, p, T):
    raise NotImplementedError

  def dv_dT_isop_pT(self, p, T):
    raise NotImplementedError

  def dp_drho_isos_pT(self, p, T):
    raise NotImplementedError
  
  def _check_and_precompute(self, p, T) -> None:
    ''' Validates cache for given p, T. If the cached value is stale,
    precomputes molar volume and energies at p, T.'''
    
    if np.any([val is None for val in self._cache_pT]) or \
      (p - self._cache_pT[0])**2 + (T - self._cache_pT[1])**2 > self.cache_tol**2:
      # Cache is stale; update is needed.
      self._cache_pT = (p, T)
    else:
      # Use cached values
      return

    ''' Compute values '''
    # Unpack
    R, V_c = self.R_mol, self.V_c
    # Pa -> MPa -> bar
    p = p * 1e-6 * 1e1
    # Precompute power terms
    powers_T_Celsius = [1, (T-273.15), (T-273.15)**2, (T-273.15)**3]
    # Compute coefficients
    coeffs_a_Celsius = [166.8e6, -193080, 186.4, -0.071288]
    coeffs_dadT_Celsius = [-193080, 2*186.4, 3*-0.071288, 0]
    b = 14.6
    a = np.dot(coeffs_a_Celsius, powers_T_Celsius)
    dadT = np.dot(coeffs_dadT_Celsius, powers_T_Celsius)
    # Define compressibility form
    Z_fn = lambda V: V/(V-b) - 1e-1 * a / (R*T*(V+b)*np.sqrt(T))
    # Cubic form for molar density (cm^3/mol), w.r.t pressure in J/cc, T in K
    poly = [1e-1*p, -R*T, -(b*b*(1e-1*p) + R*T*b - (1e-1*a)/np.sqrt(T)),
      -(1e-1*a)*b/np.sqrt(T)]
    # Dimensional vector (poly*dimensonals -> Pa cm^9 / mol^3 == J cc^2 / mol^3)
    dimensionals = [V_c**3, V_c**2, V_c, 1]
    # Define polynomial coefficients to nondimensional cubic for molar volume (V/V_c)
    poly_scaled = np.asarray(poly) * np.asarray(dimensionals)
    poly_scaled /= poly_scaled[0]

    # Solve cubic equation for nondimensional molar volume
    V_roots = np.roots(poly_scaled)
    # Filter roots: real root, V >= b
    V = np.array([np.real(root) for root in V_roots
      if np.isreal(root) and root*V_c >= b])
    # Filter roots: take gas-like for subcritical conditions
    is_using_sub_crit_filter = False
    if len(V) > 1:
      is_using_sub_crit_filter = True
      V = V[np.argmin(np.abs(Z_fn(V*V_c)-1))]
    # Cast possible array to float (np.array, float -> float, raise for list)
    V = float(V)

    # Reconstruct pressure for verification
    # p_test = [(R*T/(root*V_c - b) - 1e-1*a / (root*V_c*(root*V_c+b)*np.sqrt(T))).astype(str) + " MPa" for root in V]
    p_test = (R*T/(V*V_c - b) - 1e-1*a / (V*V_c*(V*V_c+b)*np.sqrt(T))).astype(str) + " MPa"

    # Cache compressibility factor
    Z = 1e-1 * p*(V*V_c) / (R*T)
    self._Z = Z 

    ''' Compute departure functions. '''
    # Molar departure quantities divided by RT
    #   (conversion bar * cc / mol == 1e-1 J / mol)
    #   (b has units: (cc/mol)^2 * T^(1/2) * bar)
    #   (b has units of V*V_c: cc/mol)
    # Compute departure quantities (Q - Q{ig}) / RT
    e_dep_nondim = lambda T: 1e-1 * (1.5*a - dadT*T)/ (b*R*T*np.sqrt(T)) * np.log((V*V_c)/(V*V_c+b))
    h_dep_nondim = lambda T: e_dep_nondim(T) + 1 - Z
    # Constant heat capacity model
    c_p = 3880*self.mm # 3880 J / kgK -> J / mol K
    h_ideal = self.c_p_mol * (T - self.T_c)
    e_ideal = (self.c_p_mol - R) * (T - self.T_c)
    h_total = h_ideal + R*T*h_dep_nondim
    e_total = e_ideal + R*T*e_dep_nondim

    # Compute mass-specific volume (m^3/kg)
    v = 1e-6 * V * V_c / self.mm

    # Cache top values
    self._v = V
    self._Z = Z
    # Cache debug values
    self._cache_vals = {
      "molarSI": {
        "h_ideal": h_ideal,
        "e_ideal": e_ideal,
        "h": h_total,
        "e": e_total,
        "v": 1e-6*V*V_c,
      },
      "specific": {
        "h_ideal": h_ideal / self.mm,
        "e_ideal": e_ideal / self.mm,
        "h": h_total / self.mm,
        "e": e_total / self.mm,
        "v": v,
      },
      "Z": Z,
      "debug": {
        "Z_check": Z_fn(V*V_c),
        "a": a,
        "b": b,
        "V_c": V_c,
        "rho_c": self.rho_c,
        "c_p_molar": c_p,
        "V": V,
        "poly_scaled": poly_scaled,
        "V_roots": V_roots,
        "p_test": p_test,
        "is_using_sub_crit_filter": is_using_sub_crit_filter,
      }
    }
  

class IAPWSWater(ThermodynamicMaterial):
  pass


class MixtureMeltCrystalWaterAir():
  ''' Mixture wrapper '''

  def __init__(self, k=5e-6, n=0.5):
    # Exsolved air properties
    self.air = IdealGas(c_v=718, c_p=718+287)
    # Exsolved water properties
    self.waterEx = IdealGas(c_v=1826.6237513873475, c_p=2288.0)
    # Melt + Crystal + Dissolved water properties
    self.magma = MagmaLinearizedDensity(c_v=3000, rho0=2700, K=10e9, p_ref=5e6)
    # Vector of phases
    self.phases = [self.air, self.waterEx, self.magma]
    # Solubility
    self.k = k
    self.n = n

  ''' Define solubility methods. '''

  def x_sat(self, p):
    ''' Saturation mass concentration (mass dissolved / mass liquid melt). '''
    # Henry's law
    return self.k * np.abs(p) ** self.n

  def ddp_x_sat(self, p):
    ''' Derivative of saturation mass concentration w.r.t. pressure. '''
    # d/dp Henry's law
    return self.n * self.k * np.abs(p) ** (self.n-1.0)

  def sat_indicator(self, p, ywt, yc):
    ''' Indicator function for saturated state. '''
    return float(self.x_sat(p) <= ywt / (1-ywt-yc)) if ywt < 1.0 else float(True)

  def y_wv_eq(self, p, ywt, yc):
    ''' Water volume fraction at equilibrium. ''' 
    return np.clip(ywt - self.x_sat(p)*(1-ywt-yc), 0, 1)

  def _legacy_y_wv_eq(self, p, ywt):
    ''' Water volume fraction at equilibrium. ''' 
    return np.clip(1 - (1+self.x_sat(p))*(1-ywt), 0, 1)

  def dy_wv_eq_dp(self, p, ywt, yc):
    ''' Derivative of water volume fraction w.r.t. pressure. '''
    return (ywt+yc-1) * self.ddp_x_sat(p) * self.sat_indicator(p, ywt, yc)

  ''' Define mixture methods. '''

  def volfrac(self, p, T, yA, yWv, yM):
    ''' Phase volume fractions. '''
    phi = self.vf_g(p, T, yA, yWv, yM)
    # Ideal gas partition by mole fraction
    vf = phi * self.ideal_molfrac(yA, yWv, yM)
    vf[2] = 1 - phi
    return vf

  def ideal_molfrac(self, yA, yWv, yM):
    ''' Mole fractions of ideal gases in mixture. '''
    x = np.array([yA * self.air.R, yWv * self.waterEx.R, 0])
    return x / x.sum()

  def v_mix(self, p, T, yA, yWv, yM):
    ''' Mixture specific volume '''
    return yA * self.air.v_pT(p, T) \
      + yWv * self.waterEx.v_pT(p, T) \
      + yM * self.magma.v_pT(p, T) \

  def T_ph(self, p, h, yA, yWv, yM):
    ''' Temperature as a function of mixture enthalpy. '''
    c_p_mix = self.c_p(yA, yWv, yM)
    # Sensible temperature-dependent part of enthalpy
    hs_T = h - yM * (self.magma.e_dfm(p) + p * self.magma.v_pT(p, None))
    return hs_T / c_p_mix
  
  def c_v(self, yA, yWv, yM):
    ''' Mixture heat capacity at constant pressure. '''
    return yA * self.air.c_v + yWv * self.waterEx.c_v \
      + yM * self.magma.c_v

  def c_p(self, yA, yWv, yM):
    ''' Mixture heat capacity at constant pressure. '''
    return yA * self.air.c_p + yWv * self.waterEx.c_p \
      + yM * self.magma.c_p

  def dT_dp(self, p, h, yA, yWv, yM):
    ''' Partial derivative of T(p, h, y) at constant h, y.'''
    c_p_mix = self.c_p(yA, yWv, yM)
    # Equivalent form to:
    #   return - yM / c_p_mix * (self.magma.e_dfm.ddp(p)
    #     + self.magma.v_pT(p, None)
    #     + p * self.magma.dv_dp_isoT_pT(p, None))
    return - yM / c_p_mix * self.magma.v_pT(p, None)

  def dT_dh(self, p, h, yA, yWv, yM):
    ''' Partial derivative of T(p, h, y).'''
    return 1.0 / self.c_p(yA, yWv, yM)

  def dT_dy(self, p, T, yA, yWv, yM):
    ''' Partial derivative of T(p, h, y). Note that T is passed in argument.''' 
    c_p_mix = self.c_p(yA, yWv, yM)
    # (h_m - h_wv) / c_p
    return (self.magma.e_dfm(p) + p * self.magma.v_pT(p, None) \
      - (self.waterEx.c_p - self.magma.c_v) * T) / c_p_mix
  
  def dv_dp(self, p, T, yA, yWv, yM):
    ''' Partial derivative of v(p, h, y). Note that T is passed in argument.''' 
    return yM * self.magma.dv_dp_isoT_pT(p, None) \
      + (yA * self.air.v_pT(p,T) + yWv * self.waterEx.v_pT(p,T)) * (
        self.dT_dp(p, None, yA, yWv, yM) / T - 1.0 / p)
      # + \sum_gas y_i v_i * dZ/dp / Z
  
  def dv_dh(self, p, T, yA, yWv, yM):
    ''' Partial derivative of v(p, h, y). Note that T is passed in argument.''' 
    return (yA * self.air.v_pT(p,T) + yWv * self.waterEx.v_pT(p,T)) * (
      1.0 / T) * self.dT_dh(None, None, yA, yWv, yM)
      # + \sum_gas y_i v_i * dZ/dT / Z * dT/dh
  
  def dv_dy(self, p, T, yA, yWv, yM):
    ''' Partial derivative of v(p, h, y). Note that T is passed in argument.''' 
    return (yA * self.air.v_pT(p,T) + yWv * self.waterEx.v_pT(p,T)) * (
      1.0 / T) * self.dT_dy(p, T, yA, yWv, yM) \
      + (self.waterEx.v_pT(p,T) - self.magma.v_pT(p,T))
      # + \sum_gas y_i v_i * dZ/dT / Z * dT/dy

  def vf_g(self, p, T, yA, yWv, yM):
    ''' Ideal vapour volume fraction in binary mixture. '''
    # Mixture volume
    v = yA * self.air.v_pT(p, T) \
      + yWv * self.waterEx.v_pT(p, T) \
      + yM * self.magma.v_pT(p, T)
    return (yA * self.air.v_pT(p, T)
      + yWv * self.waterEx.v_pT(p, T)) / v

  def h_mix(self, p, T, yA, yWv, yM):
    ''' Mixture enthalpy. '''
    return yA * self.air.h_pT(p, T) \
      + yWv * self.waterEx.h_pT(p, T) \
      + yM * self.magma.h_pT(p, T)

  def e_mix(self, p, T, yA, yWv, yM):
    ''' Mixture energy. '''
    return yA * self.air.e_pT(p, T) \
      + yWv * self.waterEx.e_pT(p, T) \
      + yM * self.magma.e_pT(p, T)

  def Gamma_mix(self, yA, yWv, yM):
    ''' Heat capacity ratio for the mixture. '''
    return self.c_p(yA, yWv, yM) / self.c_v(yA, yWv, yM)

  def dv_dp_s(self, p, T, yA, yWv, yM):
    ''' Isentropic derivative dv/dp '''
    y = np.array([yA, yWv, yM])
    # Partials of phasic volume in Gibbs (p,T) coordinates
    # Note: for Helmholtz variables (v, T), one can obtain vT = - pT/pv
    # from the cyclic relation, and vp = 1/pv
    vp = np.array([phase.dv_dp_isoT_pT(p,T) for phase in self.phases])
    vT = np.array([phase.dv_dT_isop_pT(p,T) for phase in self.phases])
    return np.dot(y, vp) + \
      T * np.dot(y, vT) ** 2 / (self.c_v(yA, yWv, yM) - T*np.dot(y, vT*vT/vp))

  def sound_speed(self, p, T, yA, yWv, yM):
    ''' Mixture sound speed. '''
    v = self.v_mix(p, T, yA, yWv, yM)
    return np.sqrt(-v*v / self.dv_dp_s(p, T, yA, yWv, yM))

  @staticmethod
  def T_isentropic(p, p0, T0, Gamma):
    ''' Return T(p) connected to another (p, T) through a Gamma-isentrope. '''
    return T0 * (p/p0)**((Gamma-1)/Gamma)


if __name__ == "__main__":
  ''' Unit test '''
  mixture = MixtureMeltCrystalWaterAir()
  p = 100e6
  T = 1000
  yA, yWv, yM = 0.01, 0.01, 0.98
  yWt = 0.04
  yC = 0.10
  # Liquid dependent
  yL = yM - yWt - yC
  assert(yL >= 0)
  mixture.x_sat(p)
  mixture.ddp_x_sat(p)
  mixture.sat_indicator(p, yWt, yC)

  mixture.y_wv_eq(p, yWt, yC)
  mixture.dy_wv_eq_dp(p, yWt, yC)

  v = mixture.v_mix(p, T, yA, yWv, yM)
  h = mixture.h_mix(p, T, yA, yWv, yM)
  T_check = mixture.T_ph(p, h, yA, yWv, yM)

  dTdp = mixture.dT_dp(p, h, yA, yWv, yM)
  dTdh = mixture.dT_dh(p, h, yA, yWv, yM)
  dTdy = mixture.dT_dy(p, T, yA, yWv, yM)
  vf_g = mixture.vf_g(p, T, yA, yWv, yM)
  e_mix = mixture.e_mix(p, T, yA, yWv, yM)
  Gamma = mixture.Gamma_mix(yA, yWv, yM)

  p_target = 10e6
  T_target = mixture.T_isentropic(p_target, p, T, Gamma)

  print("Outputs: ")
  print({
    "v": v,
    "h": h,
    "T": T,
    "T_check": T_check,
    "dTdp": dTdp,
    "dTdh": dTdh,
    "dTdy": dTdy,
    "vf_g": vf_g,
    "e_mix": e_mix,
    "Gamma": Gamma,
    "p_target": p_target,
    "T_target": T_target,
  })
