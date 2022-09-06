import numpy as np
import scipy.integrate

'''
------------------------
Static functions
------------------------
Atomics that map state variables to state variables
'''

def rho(arhoVec):
  return np.sum(arhoVec, axis=-1, keepdims=True)

def massfrac(arhoVec):
  return arhoVec / np.sum(arhoVec, axis=-1, keepdims=True)

def c_v(densityPerAny, physics):
  ''' Compute mixture c_v (heat capacity at constant volume)
  Input arhoVec -> c_v per unit volume
  Input massfracVec -> c_v per unit mass '''
  return (densityPerAny[...,0:1] * physics.Gas[0]["c_v"]
    + densityPerAny[...,1:2] * physics.Gas[1]["c_v"] \
    + densityPerAny[...,2:3] * physics.Liquid["c_m"])

def c_p(densityPerAny, physics):
  ''' Compute mixture c_p (heat capacity at constant volume)
  Input arhoVec -> c_p per unit volume
  Input massfracVec -> c_p per unit mass '''
  return (densityPerAny[...,0:1] * physics.Gas[0]["c_p"]
    + densityPerAny[...,1:2] * physics.Gas[1]["c_p"] \
    + densityPerAny[...,2:3] * physics.Liquid["c_m"])

def Gamma(arhoVec, physics):
  return (c_p(arhoVec, physics) / c_v(arhoVec, physics))

def velocity(mom, rho):
  return mom / rho

def mixture_spec_vol(massfracVec, p, T, physics):
  Tdivp = T / p
  return (massfracVec[...,0:1] * physics.Gas[0]["R"]
    + massfracVec[...,1:2] * physics.Gas[1]["R"]) * Tdivp \
    + massfracVec[...,2:3]/(
      physics.Liquid["rho0"] * (1.0 + 
        (p - physics.Liquid["p0"]) / physics.Liquid["K"]))

def mixture_density(massfracVec, p, T, physics):
  return 1.0 / mixture_spec_vol(massfracVec, p, T, physics)

def temperature(arhoVec, mom, e, physics):
  kinetic = 0.5*np.sum(mom*mom, axis=2, keepdims=True) / rho(arhoVec)
  return (e - arhoVec[...,2:3] * physics.Liquid["E_m0"] - kinetic) \
    / c_v(arhoVec, physics)

def gas_volfrac(arhoVec, T, physics):
  ''' Compute volume fraction of sum of gases (also called porosity). '''
  # Define useful quantities: additive constant to magma EOS
  sym1 = physics.Liquid["K"] - physics.Liquid["p0"]
  # Define partial pressure of gas mixture
  sym2 = (arhoVec[...,0:1] * physics.Gas[0]["R"]
    + arhoVec[...,1:2] * physics.Gas[1]["R"]) * T
  # Define negative b (from quadratic formula)
  b = (sym1 - sym2 - 
    physics.Liquid["K"] / physics.Liquid["rho0"] * arhoVec[...,2:3])
  return 0.5 / sym1 * (b + np.sqrt(b*b + 4*sym1*sym2))

def pressure(arhoVec, T, gas_volfrac, physics):
  return arhoVec[...,0:1] * physics.Gas[0]["R"] * T \
    + arhoVec[...,1:2] * physics.Gas[1]["R"] * T \
    + (1.0-gas_volfrac)*(physics.Liquid["p0"] - physics.Liquid["K"]) \
      + (physics.Liquid["K"] / physics.Liquid["rho0"]) * arhoVec[...,2:3]

def Psi1(p, gas_volfrac, physics):
  ''' Compute intermediate product for one computation path for sound speed. '''
  return (p + physics.Liquid["K"] - physics.Liquid["p0"]) / ( 
  p + gas_volfrac * (physics.Liquid["K"] - physics.Liquid["p0"]))

def internal_energy_per_vol(arhoVec, mom, e):
  return e - 0.5*np.sum(mom*mom, axis=2, keepdims=True)/rho(arhoVec)

def total_enthalpy_per_mass(arhoVec, p, e):
  return (e+p) / rho(arhoVec)

def sound_speed(Gamma, p, rho, gas_volfrac, physics):
  # Compute sum representation for Gamma / (rho a^2)
  summed = gas_volfrac / p + (1.0 - gas_volfrac)/(
    p - physics.Liquid["p0"] + physics.Liquid["K"])
  return np.sqrt(Gamma / summed / rho)

def volfrac_air(arhoVec, gas_volfrac, physics):
  ''' Compute air volume fraction from partial pressure partition of the
  total gas volume fraction. Fills in to avoid zero divided by zero. '''
  _out_val = np.zeros_like(gas_volfrac)
  # Compute partial pressure (divided by temperature) masked to nonzero values
  ppA = arhoVec[...,0:1][np.where(gas_volfrac != 0)] * physics.Gas[0]["R"]
  ppWv = arhoVec[...,1:2][np.where(gas_volfrac != 0)] * physics.Gas[1]["R"]
  _out_val[np.where(gas_volfrac != 0)] = \
    gas_volfrac[np.where(gas_volfrac != 0)] * ppA / (ppA + ppWv)
  return _out_val

def volfrac_water(arhoVec, gas_volfrac, physics):
  ''' Compute water volume fraction from partial pressure partition of the
  total gas volume fraction. Fills in to avoid zero divided by zero. '''
  _out_val = np.zeros_like(gas_volfrac)
  # Compute partial pressure (divided by temperature) masked to nonzero values
  ppA = arhoVec[...,0:1][np.where(gas_volfrac != 0)] * physics.Gas[0]["R"]
  ppWv = arhoVec[...,1:2][np.where(gas_volfrac != 0)] * physics.Gas[1]["R"]
  _out_val[np.where(gas_volfrac != 0)] = \
    gas_volfrac[np.where(gas_volfrac != 0)] * ppWv / (ppA + ppWv)
  return _out_val

def volfrac_magma(gas_volfrac):
  return 1.0 - gas_volfrac

def dim_entropy_per_mass(T, p, massfracVec, physics):
  ''' Compute dimensional physical entropy with respect to unit reference
  temperature and unit reference pressure. This physical entropy is not
  typically used directly; the use of isentropic relation
    T^Gamma / p^{Gamma-1} 
  is recommended. Also c_v_mix * log(T*Gamma / p^{Gamma-1}) '''
  gammaA = physics.Gas[0]["gamma"]
  gammaWv = physics.Gas[1]["gamma"]
  return massfracVec[...,0:1] * physics.Gas[0]["c_p"] * np.log(T / p**((gammaA-1.0)/gammaA)) + \
        massfracVec[...,1:2] * physics.Gas[1]["c_p"] * np.log(T / p**((gammaWv-1.0)/gammaWv)) + \
        massfracVec[...,2:3] * physics.Liquid["c_m"] * np.log(T)

def entropic_ratio_per_mass(T, p, Gamma):
  ''' Compute quantity
    T^Gamma / p^{Gamma-1}. '''
  return T^Gamma / p^(Gamma-1)

def acousticRI_integrand_scalar(p, T0, p0, massfracVec, Gamma, physics):
  ''' Optimized integrand for Riemann invariant psi where
    dpsi = du + f(p) dp
  for inputs of shape (nq,) or (1,...,1,nq,) where nq is the number of
  quadrature points (from a possibly adaptive algorithm).
  '''
  p = p.squeeze()
  T0 = T0.squeeze()
  p0 = p0.squeeze()
  Gamma = Gamma.squeeze()
  am2 = physics.Liquid["K"] / physics.Liquid["rho0"]
  yR_g = massfracVec[...,0].squeeze() * physics.Gas[0]["R"] \
    + massfracVec[...,1].squeeze() * physics.Gas[1]["R"]
  rhom_am2 = p - physics.Liquid["p0"] + physics.Liquid["K"]
  return np.sqrt((
    yR_g * T0 * (p/p0)**((Gamma-1.0)/Gamma) / (p*p)
    + massfracVec[...,2].squeeze() * am2 / (rhom_am2*rhom_am2)
  ) / Gamma)

def velocity_RI_fixed_p_quadrature(p_bdry:float, U:np.array,
  physics, normals, is_adaptive=True, tol=1e-1, rtol=1e-5) -> tuple:
  ''' Quadrature of inverse acoustic impedance for fixed integration limits:
     p                
      2               
      ⌠        -1     
      ⌡ (ρ ⋅ c)   ⋅ dp
     p                
      1               

  Result is equal to the difference in particle velocity for two phases with
  the same value of acoustic Riemann invariant.
  Returns tuple of primitives (pHat, uHat, THat)
  '''

  arhoVec = U[:,:,physics.get_mass_slice()]
  mom = U[:,:,physics.get_momentum_slice()]
  e = U[:,:,physics.get_state_slice("Energy")]

  massfracVec = massfrac(arhoVec)
  T0 = temperature(arhoVec, mom, e, physics)
  p0 = pressure(arhoVec, T0, gas_volfrac(arhoVec, T0, physics), physics)
  Gamma0 = Gamma(arhoVec, physics)

  # Set boundary pressure equal to specified pressure function
  pHat = p_bdry
  # Set initial value for velocity, updated with subsequent quadrature
  uHat = velocity(mom, rho(arhoVec))

  ''' Adaptive Gauss quadrature '''
  if is_adaptive:
    val, err = scipy.integrate.quadrature(
      lambda p: acousticRI_integrand_scalar(
        p, T0, p0, massfracVec, Gamma0, physics),
      p_bdry, p0, tol=tol, rtol=rtol)
    uHat += normals * val
  else:
    raise NotImplementedError("Non-adaptive algorithm is deprecated.")
    ''' Full-stride Gauss quadrature for ODE of type dF/dp = f(p)
    Fixed performance (too accurate for perturbations near the boundary, but
    not accurate enough for large amplitude waves). '''
    
    x_nodes = np.array([-1, -np.sqrt(3/7), 0, np.sqrt(3/7), 1])
    w = np.array([1/10, 49/90, 32/45, 49/90, 1/10])
    # Affine mapping
    p_nodes = np.expand_dims(0.5*(p0 + p_bdry), axis=-1) \
      + np.einsum("..., k -> ...k", 0.5*(p0 - p_bdry), x_nodes)
    # Define function f(p)
    f = lambda p: _acousticRI_integrand(U, p_nodes, solver.physics,
      T0, p0, y1, y2, y3, Gamma0, is_p_preshaped=True)
    # Perform Gauss quadrature
    #   Multiply f, weights in [-1, 1], jacobian of affine map [-1,1]->[p_bdry, p0]
    uHat += np.einsum("...k, k, ...", f(p_nodes), w, 0.5*(p0 - p_bdry))

  ''' Postprocess solution '''
  THat = T0 * (pHat / p0) **((Gamma0-1.0)/Gamma0)
  return(pHat, uHat, THat)