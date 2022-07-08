# ------------------------------------------------------------------------ #
#
#       Compute hydrostatic initial condition for barotropic description of
#       the conduit.
#
# ------------------------------------------------------------------------ #

from cmath import e
import numpy as np

import numerics.helpers.helpers as helpers
import meshing.tools as mesh_tools
import solver.tools as solver_tools
import scipy
import copy
from scipy.sparse import dok_array
from physics.multiphasevpT.functions import GravitySource

class GlobalDG():
  def __init__(self, solver):
    self.solver = solver
    self.origin_state = solver.state_coeffs

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
  
  def eval_barotropic_state(self, p:np.array):
    ''' Evaluate state with N-1 constraints and known pressure:
    * (4 states) arhoA, arhoWt, arhoC, xmomentum are held constant
    * (3 states) arhoWv, arhoM, e may vary subject to two constraints:
      1. T held constant
      2. rhoWv, rhoM at equilibrium dissolved concentration 
    These 6 constraints allow the state to be parametrized by pressure p.
    Given p, the algorithm is:
      0. Compute T, arhoWt (total water) from origin state
      1. Compute solubility
        arhoWd/(arhoM - arhoWd) == k*p^n
        arhoWd == arhoWt - arhoWv
        implying arhoM == arhoWd * (1/(k*p^n) + 1) == (arhoWt - arhoWv) * (1/(k*p^n) + 1)
      2. Compute phasic densities
        rhoWv == p / (R_wv * T)
        rhoM == rho0 + (rho0 / K) * (p - p0)
        rhoA == p / (R_a * T)
      3. Compute air volume fraction
        alphaA = arhoA / rhoA
        implying alpha_m = (1 - alpha_a) - alpha_wv
      4. Resubstitute into solubility equation for affine and solve
        alpha_m * rhoM == (arhoWt - alpha_wv*rhoWv) * (1/(k*p^n) + 1)
    
    Input: p in vector form
    Output: state in Quail's format (ne, nb, ns)
    '''
    physics = self.solver.physics
    iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC = \
      physics.get_state_indices()
    
    # Copy origin state
    constrained_state = self.origin_state
    # Rearrange p to Quail format
    p = self.inv_ravel(p)
    
    # Constrained states
    T = self.solver.physics.compute_additional_variable(
      "Temperature", constrained_state, flag_non_physical=True)
    arhoWt = self.origin_state[:,:,physics.get_state_slice("pDensityWt")]

    conc_eq = physics.Solubility["k"] * p ** physics.Solubility["n"]
    rhoA = p / (physics.Gas[0]["R"]*T)
    rhoWv = p / (physics.Gas[1]["R"]*T)
    rhoM = physics.Liquid["rho0"] + (physics.Liquid["rho0"] 
      / physics.Liquid["K"]) * (p - physics.Liquid["p0"])
    alphaA = self.origin_state[:,:,physics.get_state_slice("pDensityA")] / rhoA
    alphaWv = (arhoWt * conc_eq - (1.0 - alphaA) * rhoM) \
      / (conc_eq * rhoWv - rhoM)
    alphaM = 1.0 - alphaA - alphaWv

    # Compute volumetric energy
    e = alphaA * rhoA * physics.Gas[0]["c_v"] * T \
      + alphaWv * rhoWv * physics.Gas[1]["c_v"] * T \
      + alphaM * rhoM * (physics.Liquid["c_m"] * T + physics.Liquid["E_m0"])

    constrained_state[:,:,physics.get_state_slice("pDensityWv")] = alphaWv * rhoWv
    constrained_state[:,:,physics.get_state_slice("pDensityM")] = alphaM * rhoM
    constrained_state[:,:,physics.get_state_slice("Energy")] = e
   
    return constrained_state

  def set_initial_condition(self, p_bdry:float=1e5, p_bc_loc:str="x2"):
    ''' Replace the initial condition in self.solver with hydrostatic
    condition.
    
    p_bdry: pressure at boundary
    p_bc_loc: location of pressure boundary condition
    '''

    solver = self.solver

    ''' Identify index and normal corresponding to bc loc '''
    filtered = [i for i in range(self.N) 
      if np.isclose(self.x[i], 
        solver.mesh.node_coords[
          solver.mesh.elem_to_node_IDs[
            solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].elem_ID]
          [solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].face_ID]])]
    if len(filtered) != 1:
      raise Exception("Multiple nodes at to specified pressure BC location.")
    BC_index = filtered[0]
    BC_normal_sign = solver.elem_helpers.normals_elems[
      solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].elem_ID,:,0,0][
        solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].face_ID]

    ''' Construct boundary load vector '''
    b = dok_array((self.N, 1,))
    b[BC_index] = BC_normal_sign * 0.5 * p_bdry
    b = b.tocsr()

    ''' Assemble local Lax-Friedrichs numerical flux matrix A '''
    A = dok_array((self.N, self.N,))
    nb = self.nb
    for i in range(self.ne):
        A[nb*i, nb*i] = -0.5
        A[nb*(i+1)-1, nb*(i+1)-1] = 0.5
        if i > 0:
            A[nb*i, nb*i-1] = -0.5
        if i < self.ne-1:
            A[nb*(i+1)-1, nb*(i+1)] = 0.5
    # Check for one-sided numerical fluxes where BC is absent
    if BC_index == 0:
      A[-1,-1] = 1.0
    elif BC_index == self.N-1:
      A[0,0] = -1.0
    A = A.tocsr()

    ''' Assemble interior flux matrix B '''
    u = np.einsum('jn, jm, ijm -> ijn', 
      solver.basis.get_values(solver.elem_helpers.quad_pts),
      solver.elem_helpers.quad_wts,
      solver.elem_helpers.djac_elems)
    # [ne, nq, nb1, ndims] x [ne, nq, nb2] -> [ne, nb1, nb2]
    B_vec = np.einsum('ijml, ijn -> imn',
      solver.elem_helpers.basis_phys_grad_elems,
      u)
    B = dok_array((self.N, self.N,))
    nb = self.nb
    for i in range(self.ne):
      B[nb*i:nb*(i+1), nb*i:nb*(i+1)] = B_vec[i,:,:]
    B = B.tocsr()

    Uq = helpers.evaluate_state(solver.state_coeffs, solver.elem_helpers.basis_val,
            skip_interp=solver.basis.skip_interp)

    ''' Assemble interior mass matrix M '''
    # [nq, nb] x [nq, 1] x [ne, nq, 1] -> [ne, nq, nb]
    u = np.einsum('jn, jm, ijm -> ijn', 
      solver.basis.get_values(solver.elem_helpers.quad_pts),
      solver.elem_helpers.quad_wts,
      solver.elem_helpers.djac_elems)
    # [nq, nb1] x [ne, nq, nb2] -> [ne, nb1, nb2]
    M_vec = np.einsum('jm, ijn -> imn',
      solver.basis.get_values(solver.elem_helpers.quad_pts),
      u)
    # TODO: Looks like mass matrix here is 1/2 as large as it should be? But it
    # works. The following
    # gives the integral over [-1,1] (segment with length 2, rather than 1):
    # M_vec[0,:,:] / (gdg.x[1] - gdg.x[0])
    M = dok_array((self.N, self.N,))
    nb = self.nb
    for i in range(self.ne):
      M[nb*i:nb*(i+1), nb*i:nb*(i+1)] = M_vec[i,:,:]
    M = M.tocsr()

    ''' Compute initial guess from average weight '''
    gsource = [s for s in solver.physics.source_terms if
      type(s) is GravitySource][0]
    # Barotropic gravity source function [ne, nq]
    S_fn = lambda p: gsource.get_source(
        self.solver.physics,
        self.eval_barotropic_state(p),
        self.solver.elem_helpers.x_elems,
        self.solver.time)[
          :,:,self.solver.physics.get_state_index("XMomentum")]
    vec_cast = lambda S: np.expand_dims(S.ravel(),axis=1)
    # Load vector
    f_fn = lambda p: -b + M @ vec_cast(S_fn(p))
    # # Calculate source term quadrature [ne, nq, ns]
    # Sq_quad = np.einsum('ijk, jm, ijm -> ijk', 
    #     Sq, 
    #     solver.elem_helpers.quad_wts,
    #     solver.elem_helpers.djac_elems)

    ''' Compute initial guess '''
    rho0 = np.sum(
      solver.state_coeffs[:,:,solver.physics.get_mass_slice()],axis=2) \
      .ravel()[BC_index]
    p_like = 0.0*b.todense() + 1.0
    f = -b + (-rho0*gsource.gravity) * M @ p_like
    p = scipy.sparse.linalg.spsolve(A-B, f)

    ''' Fixed point iteration '''
    fixedpointiter = lambda p: scipy.sparse.linalg.spsolve(A-B,
      f_fn(np.expand_dims(p,axis=1)))

    # Residual in algebraic equation
    evalresidual = lambda p : np.linalg.norm(
      (A-B)@np.expand_dims(p,axis=1) - f_fn(np.expand_dims(p,axis=1)), 'fro')
    N_iter = 30
    residuals = np.zeros(N_iter)
    for i in range(N_iter):
      p = fixedpointiter(p)
      residuals[i] = evalresidual(p)
    
    # Replace solver state coefficients with hydrostatic state
    # self.solver.state_coeffs = self.eval_barotropic_state(p)