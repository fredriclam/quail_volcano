''' Adapter for parallel steady state exploration '''


import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import steady_state as ss

from types import MappingProxyType

standard_properties = {
  "yC": 0.4,
  "yWt": 0.05,
  "yA": 1e-7,
  "yWvInletMin": 1e-7,
  "crit_volfrac": 0.8,
  "tau_d": 1.0e-1,
  "tau_f": 1.0e-1,
  "conduit_radius": 50,
  "T_chamber": 1000,
  "c_v_magma": 3e3,
  "rho0_magma": 2.7e3,
  "K_magma": 10e9,
  "p0_magma": 5e6,
  "solubility_k": 5e-6,
  "solubility_n": 0.5,
  "crit_strain_rate_k": 0.01,
  "fragmentation_criterion": "StrainRate", # VolumeFraction
}
# Set default properties from a copy of standard properties
# MappingProxyType for read-only is not picklable
_default_properties = standard_properties.copy()

# print("Preparing module level steady state object...")

# Set arbitrary values for init (using interface compatible with Quail)
x = np.linspace(-1150, -150, 1000)
p_vent = 1e5
u_inlet = 1

# print("Done preparing module level steady state object.")

def parallel_compute_qoi(p_mg, j_mg, props:dict=_default_properties, pool_size:int=24) -> np.array:
  # Pack p, j, props as list of tuples
  arg_list = [(p, j, props) for p, j in zip(p_mg.ravel(), j_mg.ravel())]
  out = None
  with mp.Pool(processes=pool_size) as pool:
    print("Set up pool.")
    print(pool)
    out = list(pool.starmap(compute_qoi, arg_list))
  if out is None:
    raise ValueError("Starmap failed to return output list in parallel_compute_qoi.")
  # Rearrange list into meshgrid shape
  return np.array(out, dtype=object).reshape(p_mg.shape)

def compute_qoi(p_chamber, j0, props:dict=_default_properties) -> dict:
  ''' Computes dict of quantities of interest within the conduit from
  the solution using pressure as an independent coordinate.'''

  f = ss.SteadyState(x, p_vent, u_inlet, input_type="u",
    override_properties=props, skip_rootfinding=True
  )
  # Call steady-state solve (bottom boundary only)
  sol_obj:dict = f.solve_pcoord_system(p_chamber, j0)

  # Compute all quantities dependent on p
  _p = np.linspace(*sol_obj["range(p)"], 1000)
  _x = sol_obj["x(p)"](_p).ravel()
  _h, _yw, _yf = sol_obj["hyy(p)"](_p)
  _T = f.T_ph(_p, _h, _yw)
  # Specific volume
  _v = f.mixture.v_mix(_p, _T, f.yA, _yw, 1 - f.yA - _yw)
  _rho = 1.0 / _v
  # Viscosity
  _mu = f.F_fric_viscosity_model(_T, _yw, _yf)
  # Vol frac vector (N,3)
  vf = np.array(list(map(f.mixture.volfrac, _p, _T, f.yA*np.ones_like(_yw), _yw, 1 - f.yA - _yw)))
  # Strain rate
  strain_rate = [f.strain_rate(scalar_p, scalar_T, scalar_y, scalar_yF, j0)
                for scalar_p, scalar_T, scalar_y, scalar_yF
                  in zip(_p, _T, _yw, _yf)]
  # Sound speed
  _c = np.array(list(map(f.mixture.sound_speed, _p, _T, f.yA * np.ones_like(_p), _yw, 1.0 - (f.yA + _yw))))
  # Mach number
  _M = j0 * _v / _c
  # Correction for strain rate at sonic boundary (extrapolate)
  strain_rate[0] = strain_rate[1] \
    + (strain_rate[2] - strain_rate[1]) / (_p[2] - _p[1]) * (_p[0] - _p[1])
  # Distribution of critical strain rate
  crit_strain_rate = f.crit_strain_rate_k * f.K_magma / _mu

  def compute_front_loc(yf:np.array, x:np.array, value_at_front=0.9) -> float:
    ''' Compute front location by searching from lowest-pressure side (top). '''
    # Compute x difference
    _i = np.argmax(yf < value_at_front)
    _j = np.clip(_i-1, 0, None).astype(int)
    # Linearly interpolate
    _loc = np.interp(value_at_front,
                    yf[[_i, _j]],
                    x[[_i, _j]])
    return _loc

  # Location of % fragmentation
  # Avoids global interpolate like
  #   frag_height:float = np.interp(0, (strain_rate - crit_strain_rate)[::-1], _x[::-1])
  frag_height_50:float = compute_front_loc(_yf, _x, 0.50)
  frag_height_90:float = compute_front_loc(_yf, _x, 0.90)
  frag_height_99:float = compute_front_loc(_yf, _x, 0.99)
  frag_height_999:float = compute_front_loc(_yf, _x, 0.999)

  def compute_front_vf(yf:np.array, vf:np.array, value_at_front=0.9) -> float:
    ''' Compute front location by searching from lowest-pressure side (top). '''
    # Compute x difference
    _i = np.argmax(yf < value_at_front)
    _j = np.clip(_i-1, 0, None).astype(int)
    # Linearly interpolate
    _loc = np.interp(value_at_front,
                    yf[[_i, _j]],
                    1.0 - vf[:,2][[_i, _j]])
    return _loc

  frag_vf_50:float = compute_front_vf(_yf, vf, 0.50)
  frag_vf_90:float = compute_front_vf(_yf, vf, 0.90)
  frag_vf_99:float = compute_front_vf(_yf, vf, 0.99)
  frag_vf_999:float = compute_front_vf(_yf, vf, 0.999)

  # Outflow mach number
  mach_out:float = _M[0]

  return {
    "p": _p,
    "x": _x,
    "h": _h,
    "yw": _yw,
    "yf": _yf,
    "T": _T,
    "v": _v,
    "rho": _rho,
    "u": j0 * _v,
    "mu": _mu,
    "vf": vf,
    "strain_rate": strain_rate,
    "c": _c,
    "M": _M,
    "crit_strain_rate": crit_strain_rate,
    "frag_height_50": frag_height_50,
    "frag_height_90": frag_height_90,
    "frag_height_99": frag_height_99,
    "frag_height_999": frag_height_999,
    "frag_vf_50": frag_vf_50,
    "frag_vf_90": frag_vf_90,
    "frag_vf_99": frag_vf_99,
    "frag_vf_999": frag_vf_999,
    "mach_out": mach_out,
  }
