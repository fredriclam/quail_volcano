import numpy as np
import atomics

# Input mass flux (kg/m^3 * m/s)
j = 20000

# Define test physics
class PhysicsPlaceholder():
  def __init__(self):
    self.Gas = [
      {"c_v": 718, "R": 8314/28.84, "c_p": 718 + 8314/28.84},
      {"c_v": 1e3, "R": 8314/18.02, "c_p": 1e3 + 8314/18.02},
    ]
    self.Liquid = {
      "c_m": 3e3,
      "K": 10e9,
      "p0": 5e6,
      "rho0": 2.5e3,
      "E_m0": 0,
    }
  def get_mass_slice(self):
    return slice(0,3)
  def get_momentum_slice(self):
    return slice(3,4)
  def get_state_slice(self, key):
    if key == "Energy":
      return slice(4,5)
    raise Exception
physics = PhysicsPlaceholder()
# Define test input data
U = np.zeros((1,1,7))
e = 0
T = 1000
U[...,:] = np.array([0.1*0, 2.0*1e-7, 2499, 0, e, 10, 100])
e = atomics.c_v(U[...,0:3], physics) * T
U[...,4] = e
# Define test chamber parameters
T_r = 1200
p_r = 10e6
# Define outward boundary normals
normals = -np.ones(U.shape[0:-1])

''' Computation '''
Gamma = atomics.Gamma(U[...,0:3], physics)
y = atomics.massfrac(U[...,0:3])
yRGas = y[...,0] * physics.Gas[0]["R"] + y[...,1] * physics.Gas[1]["R"]
# Entropy as ratio of T/p^n
S_r = T_r / p_r**((Gamma-1)/Gamma)
K, rho0, p0 = physics.Liquid["K"], physics.Liquid["rho0"], physics.Liquid["p0"]
uGrid = atomics.velocity(U[...,3:4], np.sum(U[...,0:3]))
TGrid = atomics.temperature(U[...,0:3], U[...,3:4], U[...,4:5], physics)
pGrid = atomics.pressure(U[...,0:3], TGrid, atomics.gas_volfrac(U[...,0:3], T, physics), physics)
# print(T, atomics.temperature(U[...,0:3], U[...,3:4], U[...,4:5], physics))
# print(atomics.gas_volfrac(U[...,0:3], T, physics))
# exit()

# Lambdas for G, dGdp
# G = lambda p: j * (yRGas * p**(-1/Gamma) * S_r + y[...,2] * K / (rho0*(p + K - p0))) \
#   - uGrid \
#   - atomics.velocity_RI_fixed_p_quadrature(p, U, physics, normals, is_adaptive=True, tol=1e-1, rtol=1e-5)
# dGdp = lambda p: j * (yRGas * (-1/Gamma) * p **(-1/Gamma) / p * S_r - y[...,2] * K / (rho0*(p + K - p0)**2.0)) \
#   - atomics.acousticRI_integrand_scalar(p, TGrid, pGrid, y, Gamma, physics)

def eval_fun_dfun(p):
  ''' Evaluate function G and its derivative dG/dp. '''
  # Define reusable groups
  g1 = yRGas * p**(-1/Gamma) * S_r
  g2 = y[...,2] * K / (rho0*(p + K - p0))
  # Integration of 1/impedance
  # Note that the output p, T are not used, since entropy is not taken from grid
  _, uTarget, _ = atomics.velocity_RI_fixed_p_quadrature(p, U, physics, normals,
    is_adaptive=True, tol=1e-1, rtol=1e-5)
  # Evaluate integrand
  f = atomics.acousticRI_integrand_scalar(np.array(p), TGrid, pGrid, y, Gamma, physics)
  # Evaluate returns
  G = j * (g1 + g2) + normals * uTarget
  dGdp = -j * (g1 * (1/Gamma) / p + g2 / (p+K-p0)) - f
  return G, dGdp, (p, uTarget)

def newton_iter(p=pGrid, verbose=False):
  G = None
  # Set Newton tolerance for residual equation (units of velocity)
  newton_tol = 1e-7
  for i in range(10):
    print(f"Iter {i}: p = {p}, residual G = {G}") if verbose else None
    G, dGdp, _ = eval_fun_dfun(p)
    p -= G / dGdp
    if np.abs(G) < newton_tol:
      break
  print(f"Iter final: p = {p}, residual G = {G}") if verbose else None
  return p

''' Solve with verbose output '''
print("Newton iteration to solve equation G = 0")
p = newton_iter(verbose=True)
print("="*10)

''' Performance timing '''
from time import perf_counter
N_tests = 500
t1 = perf_counter()
for i in range(N_tests):
  newton_iter(verbose=False)
t2 = perf_counter()
print(f"Wallclock average: {(t2-t1)/N_tests*1e3} ms")
print("="*10)

''' Verify final solution '''
_, _, (p, u) = eval_fun_dfun(p)
T = S_r * p**((Gamma-1)/Gamma)

# Mass flux of boundary state
bdry = {
  "j": atomics.mixture_density(y, p, T, physics) * u,
  "u": u,
  "T": T,
  "p": p,
}
# Condition 1: mass flux matching
cond1 = bdry["j"], j
# Condition 2: entropy matching with chamber
cond2 = bdry["T"] / bdry["p"]**((Gamma-1)/Gamma), S_r
# Condition 3: acoustic RI matching with grid
cond3 = bdry["u"] - atomics.velocity_RI_fixed_p_quadrature(bdry["p"], U, physics, normals,
    is_adaptive=True, tol=1e-1, rtol=1e-5)[1], uGrid
print("Verification: out and prescribed values should be np.isclose")
print(f"Mass flux (out, prescribed): {cond1}")
print(f"Entropy as T / p**((G-1)/G) (out, prescribed): {cond2}")
print(f"Velocity at grid pressure (out, prescribed): {cond3}")