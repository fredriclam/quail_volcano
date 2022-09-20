# Utility functions
import numpy as np
from scipy.optimize import fsolve

class RiemannSolution():
  def __init__(self, physics):
    self.physics = physics

  def wave_speed(self, U):
    ''' Compute the wave speed u - a corresponding to left-propagating expansion
    waves. '''
    return U[:,:,3]/U[:,:,0:3].sum(axis=2) - \
      self.physics.compute_variable("SoundSpeed", U)

  def path_vec(self, U):
    ''' Compute the tangential vector for the integral curve corresponding to
    the expansion wave. '''
    curve_path = np.zeros_like(U)
    curve_path[:,:,0:3] = U[:,:,0:3]/U[:,:,0:3].sum(axis=2, keepdims=True)
    curve_path[:,:,5:7] = U[:,:,5:7]/U[:,:,0:3].sum(axis=2, keepdims=True)
    curve_path[:,:,3] = self.wave_speed(U)
    return curve_path

  

  def mass_spec_entropy_mixture(self, U):
    ''' Compute the mass-specific thermodynamic entropy of the mixture. '''
    T = self.physics.compute_variable("Temperature", U)
    p = self.physics.compute_variable("Pressure", U)
    def mass_spec_entropy_gas(T, p, Gas):
      gamma = Gas["gamma"]
      return Gas["c_p"] * np.log(T / p**((gamma-1.0)/gamma))
    return (
      U[:,:,0] * mass_spec_entropy_gas(T, p, self.physics.Gas[0])[:,:,0]
      + U[:,:,1] * mass_spec_entropy_gas(T, p, self.physics.Gas[1])[:,:,0]
      + U[:,:,2] * self.physics.Liquid["c_m"] * np.log(T)[:,:,0]
      ) / U[:,:,0:3].sum(axis=2)

  def step(self, U, drho, U_origin=None):
    ''' Compute a step along the integral curve for density change drho.
    Provide U_origin for computing entropy; otherwise the entropy change
    is computed based on the current U. '''
    if U_origin is None:
      U_origin = U

    ''' Step along integral curve '''
    # Butcher table for Cash-Karp RK quadrature
    B = np.array([[1/5, 0, 0, 0, 0],
          [3/40, 9/40, 0, 0, 0],
          [3/10, -9/10, 6/5, 0, 0],
          [-11/54, 5/2, -70/27, 35/27, 0],
          [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]])
    w = np.array([37/378, 0, 250/621, 125/594 , 0, 512/1771])
    num_stages = w.shape[0]
    k = np.zeros((num_stages, *U.shape,))
    k[0,:,:,:] = self.path_vec(U)
    for j in range(B.shape[0]):
        k[j+1,:,:,:]= self.path_vec(U + drho*
            np.einsum("m, mijk -> ijk", B[j,0:j+1], k[0:j+1,:]))
    dU = drho * np.einsum("i, ijkl -> jkl", w, k)

    ''' Compute energy as dependent variable from entropy condition '''
    compute_S = lambda U : self.mass_spec_entropy_mixture(U).squeeze()
    # Construct unit vector for energy
    e_unitvec = np.zeros_like(U)
    e_unitvec[:,:,4:5] = 1.0
    # Solve isentropic energy density change
    de = fsolve(lambda de: 
      compute_S(U + dU + de*e_unitvec)/compute_S(U_origin) - 1.0, 0)
    return U + dU + de*e_unitvec

  def compute_solution(self, U_L, delta_rho=-2.0):
    ''' Compute mapping as pairs (wavespeed, U)'''
    U = U_L.copy()
    rho = U[0,0,0:3].sum()
    # Compute number of steps possible down to vacuum (rho == 0)
    ind_max = int(rho/np.abs(delta_rho))
    # Allocate for output data
    wavespeed_data = np.zeros((ind_max+1,))
    U_data = np.zeros((ind_max+1, *U.shape,))
    wavespeed_data[0] = self.wave_speed(U)
    U_data[0,:,:,:,] = U

    for i in range(ind_max):
      U = self.step(U, delta_rho, U_origin=U_L)
      rho += delta_rho
      wavespeed_data[i+1] = self.wave_speed(U)
      U_data[i+1,:,:,:] = U
    
    return wavespeed_data, U_data