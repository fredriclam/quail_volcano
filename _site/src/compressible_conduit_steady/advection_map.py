import numpy as np
import scipy.integrate

def advection_map(x:np.array, xp:np.array, up:np.array,
  source_time_functions:tuple):
  ''' Maps x to values of each source time function, advected by the velocity
  field up(xp). Input args as if using np.interp. Assumes phase at xp.min()
  is equal to zero; that is, each f in source_time_functions is evaluated at
  t = 0 at xp.min(). '''
  # Rearrange input (x,u) into linear arrays
  isort = np.argsort(xp.ravel())
  x_lin = xp.ravel()[isort]
  u_lin = up.ravel()[isort]
  # Numerically integrate slowness
  taus = scipy.integrate.cumulative_trapezoid(1.0/u_lin, x=x_lin, initial=0)
  # Set up numerical coordinate map from t -> x given velocity field
  coord_map = lambda xq: np.interp(xq, x_lin, taus)
  # Map x -> t -> source quantity
  return tuple(f(-coord_map(x)) for f in source_time_functions)