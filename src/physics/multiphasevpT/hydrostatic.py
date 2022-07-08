# ------------------------------------------------------------------------ #
#
#       Compute hydrostatic initial condition for barotropic description of
#       the conduit.
#
# ------------------------------------------------------------------------ #

import numpy as np

import numerics.helpers.helpers as helpers
import meshing.tools as mesh_tools
import solver.tools as solver_tools
import numerics.helpers.helpers as helpers
import scipy
import copy

class GlobalDG():
  def __init__(self, solver):
    self.solver = solver

    ''' Get global nodal points '''
    mesh = solver.mesh
    # Get reference element nodes
    nodal_pts = solver.basis.get_nodes(solver.order)
    # Allocate [ne] x [nb, ndims]
    xphys = np.empty((mesh.num_elems,) + nodal_pts.shape)
    for elem_ID in range(mesh.num_elems):
      # Fill coordinates in physical space
      xphys[elem_ID] = mesh_tools.ref_to_phys(mesh, elem_ID, nodal_pts)
    self.x = xphys.ravel()

    # Compute size of block corresponding to element
    self.nb = len(nodal_pts)
    self.ne = mesh.num_elems
    # Compute size of system for global weak form
    self.N = mesh.num_elems*len(nodal_pts)
    
    # Compute elementwise shape
    self.eltwise_shape  = (mesh.num_elems, self.nb, 1)
    # Compute unraveled shape
    self.vec_shape = (self.N, 1)

  def inv_ravel(self, data:np.array):
    ''' Inverse of ravel operation: (N,1) -> (ne, nb, 1) '''
    return data.reshape(self.eltwise_shape)

if __name__ == '__main__':

  gdg = GlobalDG(copy.deepcopy(solver))

  from scipy.sparse import csr_array, dok_array
  A = dok_array((gdg.N, gdg.N,))
  nb = gdg.nb
  for i in range(gdg.ne):
      A[nb*i, nb*i] = -0.5
      A[nb*(i+1)-1, nb*(i+1)-1] = 0.5
      if i > 0:
          A[nb*i, nb*i-1] = -0.5
      if i < gdg.ne-1:
          A[nb*(i+1)-1, nb*(i+1)] = 0.5
  # Wall boundary condition at bottom
  A[-1,-1] = 1
  A = A.tocsr()
  plt.spy(A)

  # Constuct boundary load vector
  p_top = 1e5
  b = dok_array((gdg.N, 1,))
  b[0] = -0.5 * p_top
  b = b.tocsr()

  u = np.einsum('jn, jm, ijm -> ijn', 
    solver.basis.get_values(solver.elem_helpers.quad_pts),
    solver.elem_helpers.quad_wts,
    solver.elem_helpers.djac_elems)
  # [ne, nq, nb1, ndims] x [ne, nq, nb2] -> [ne, nb1, nb2]
  B_vec = np.einsum('ijml, ijn -> imn',
    solver.elem_helpers.basis_phys_grad_elems,
    u)
  B = dok_array((gdg.N, gdg.N,))
  nb = gdg.nb
  for i in range(gdg.ne):
    B[nb*i:nb*(i+1), nb*i:nb*(i+1)] = B_vec[i,:,:]
  B = B.tocsr()

  Uq = helpers.evaluate_state(solver.state_coeffs, solver.elem_helpers.basis_val,
          skip_interp=solver.basis.skip_interp)

  from physics.multiphasevpT.functions import GravitySource
  gsource = GravitySource()
  Sq = gsource.get_source(
      solver.physics,
      Uq,
      solver.elem_helpers.x_elems,
      solver.time)

  # Calculate source term quadrature [ne, nq, ns]
  Sq_quad = np.einsum('ijk, jm, ijm -> ijk', 
      Sq, 
      solver.elem_helpers.quad_wts,
      solver.elem_helpers.djac_elems)
  # Calculate residual
  res_elem = np.einsum('jn, ijk -> ink', basis_val, Sq_quad) # [ne, nb, ns]
  res_elem.shape

  # [nq, nb] x [nq, 1] x [ne, nq, 1] -> [ne, nq, nb]
  u = np.einsum('jn, jm, ijm -> ijn', 
    solver.basis.get_values(solver.elem_helpers.quad_pts),
    solver.elem_helpers.quad_wts,
    solver.elem_helpers.djac_elems)
  # [nq, nb1] x [ne, nq, nb2] -> [ne, nb1, nb2]
  M_vec = np.einsum('jm, ijn -> imn',
    solver.basis.get_values(solver.elem_helpers.quad_pts),
    u)

  # Looks like mass matrix here is 1/2 as large as it should be; the following
  # gives the integral over [-1,1] (segment with length 2, rather than 1)
  M_vec[0,:,:] / (gdg.x[1] - gdg.x[0])

  # M Assembly
  M = dok_array((gdg.N, gdg.N,))
  nb = gdg.nb
  for i in range(gdg.ne):
    M[nb*i:nb*(i+1), nb*i:nb*(i+1)] = M_vec[i,:,:]
  M = M.tocsr()
  plt.spy(M)
  M[0:3,0:3].todense()

  ''' Compute initial guess from average weight '''

  rho0 = 1000
  g = 10
  p_like = 0.0*b.todense() + 1.0
  f = -b + (rho0*g) * M @ p_like
  # f = -b - (rho0*g) * M @ (1+1e-2*p.^0.5)
  p_lin = scipy.sparse.linalg.spsolve(A-B, f)

  ''' Perform fixed point iteration '''

  f_fn = lambda p : -b + (rho0*g) * M @ (1+1e-2*np.abs(p/1e2)**0.5)
  p_nonlin = p_lin
  fixedpointiter = lambda p: scipy.sparse.linalg.spsolve(A-B, f_fn(np.expand_dims(p,axis=1)))

  # Residual in algebraic equation
  evalresidual = lambda p : np.linalg.norm(
    (A-B)@np.expand_dims(p,axis=1) - f_fn(np.expand_dims(p,axis=1)), 'fro')
  N_iter = 30
  residuals = np.zeros(N_iter)
  for i in range(N_iter):
    p_nonlin = fixedpointiter(p_nonlin)
    plt.plot(gdg.x, p_nonlin, '.-')
    residuals[i] = evalresidual(p_nonlin)

  plt.xlabel("depth (m)")
  plt.xlabel("pressure (Pa)")
  plt.figure()
  plt.semilogy(residuals, '.-')
  plt.xlabel("Iter")
  plt.ylabel("Algebraic residual")
  plt.show()