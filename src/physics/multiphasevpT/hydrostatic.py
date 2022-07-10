# ------------------------------------------------------------------------ #
#
#       Compute hydrostatic initial condition for barotropic description of
#       the conduit. Defines a GlobalDG object that has reference to
#       a `solver`, and a method `set_initial_condition` to override the
#       initial condition of `solver`. Typical usage is by injection through
#       Multidomain.Domain; provide in input file the list Inject, with field
#       dict("function":GlobalDG(solver).set_initial_condition(...),
#       "Initial": True, "Postinitial": False,). 
#
# ------------------------------------------------------------------------ #

import numpy as np

import numerics.helpers.helpers as helpers
import meshing.tools as mesh_tools
import solver.tools as solver_tools
import numerics.basis.tools as basis_tools
import scipy
import processing.readwritedatafiles as readwritedatafiles
import matplotlib.pyplot as plt
import copy
import logging

from scipy.sparse import dok_array
from typing import Callable
from physics.multiphasevpT.functions import GravitySource

# Suggested improvements:
# TODO: Consider adding a tolerance on total water partial density. Useful to
#       have total water > dissolved for positivity of dissolved water during
#       the unsteady solve, at the price of adding noise in exsolution source.
# TODO: Check if overintegration perturbs the unsteady residual
#       In unsteady residual, local momentum residual > 1e1 to the right of test
#       jump due to magma partial density reaching ~2500, and then plateauing to
#       the left.
# TODO: Consider factorizations that increase FPI precision (currently plateaus
#       at 1e-8).

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
    self.xphys = xphys

    # Compute size of block corresponding to element
    self.nb = len(nodal_pts)
    self.ne = mesh.num_elems
    # Compute size of system for global weak form
    self.N = mesh.num_elems*len(nodal_pts)
    # Compute elementwise shape
    self.eltwise_shape  = (mesh.num_elems, self.nb, 1)
    # Compute unraveled shape
    self.vec_shape = (self.N, 1)

    # Fixed point iteration parameters
    self.FPI_TOL = 1e-7
    self.N_iter = 20

    # Allocate error metrics
    self.residuals = None
    self.pError = None
    self.unsteady_residuals_elements = None
    self.unsteady_residuals_face = None
    self.unsteady_residuals = None

    # Initalize logger
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)
    h = logging.FileHandler(
      filename=f"hydrostatic_{hash(solver)}.log",
      encoding="utf-8")
    h.setFormatter(logging.Formatter(
      '[%(asctime)s][%(levelname)s] Logger <%(name)s> : %(message)s'))
    self.logger.addHandler(h)

  def inv_ravel(self, data:np.array):
    ''' Inverse of ravel operation: (N,1) -> (ne, nb, 1) '''
    return data.reshape(self.eltwise_shape)
  
  def eval_barotropic_state(self, p:np.array, constr_key:str="MEq"):
    ''' Evaluate state with ns-1 constraints and known pressure. The quantities
    T, arhoA, arhoC, xmomentum provided by the unequilibrated initial condition
    are computed from the latter and held constant. For the remaining three
    quantities, two constraints are required (besides for input pressure);
    several strategies are provided, accessed by providing const_key:
      - WtEq: Fixed total water arhoWt, equilibrium dissolved water content.
        Typically difficult to use (volume fraction calculated may not be valid)
      - WvEq: Fixed exsolved water, equilibrium dissolved water content. Low
        pressures may cause problems with the magma density. Invalid volume
        fractions are clipped to [0,1].
      - MEq (default): Fixed magma, equilibrium dissolved water content. Invalid
        volume fractions are clipped to [0,1].
    
    Inputs:
      p: array of pressures in raveled vector form
      constr_key: key for constraints
    Side-effects:
      self.solver.state_coeffs (shape: ne, nb, ns) is modified to equilibrium
    '''

    physics = self.solver.physics
    iarhoA, iarhoWv, iarhoM, imom, ie, iarhoWt, iarhoC = \
      physics.get_state_indices()

    # Copy origin state
    constrained_state = self.origin_state.copy()
    # Rearrange p to Quail format
    p = self.inv_ravel(p)

    # Constrained states
    T = self.solver.physics.compute_additional_variable(
      "Temperature", constrained_state, flag_non_physical=True)
    # Compute equilibrium mass-concentration
    conc_eq = physics.Solubility["k"] * p ** physics.Solubility["n"]
    # Compute phasic densities
    rhoA = p / (physics.Gas[0]["R"]*T)
    rhoWv = p / (physics.Gas[1]["R"]*T)
    rhoM = physics.Liquid["rho0"] + (physics.Liquid["rho0"] 
      / physics.Liquid["K"]) * (p - physics.Liquid["p0"])
    alphaA = self.origin_state[:,:,physics.get_state_slice("pDensityA")] / rhoA

    # Compute alphaWv, alphaM, arhoWt based on constraints specified
    if constr_key == "WtEq":
      ''' Algorithm 1: fixed wt (invalid for some values of initial condition)
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
        '''
      arhoWt = self.origin_state[:,:,physics.get_state_slice("pDensityWt")]
      C = 1.0 + 1.0 / conc_eq
      alphaWv = (C * arhoWt - (1.0 - alphaA) * rhoM) \
        / (C * rhoWv - rhoM)
      alphaM = 1.0 - alphaA - alphaWv
    elif constr_key == "WvEq":
      ''' Algorithm 2: fixed Wv (adjust wt based on equilibrium dissolution)'''
      # Preserve water vapor content
      alphaWv = self.origin_state[:,:,physics.get_state_slice("pDensityWv")] \
         / rhoWv
      alphaM = 1.0 - alphaA - alphaWv
      # Forward computation of dissolved water content
      arhoWd = conc_eq / (1.0 + conc_eq) * alphaM * rhoM
      arhoWt = arhoWd + alphaWv * rhoWv
      # Force positivity
      alphaM[np.where(alphaM < 0)] = 0.0
      alphaWv = 1.0 - alphaA - alphaM
    elif constr_key == "MEq":
      ''' Algorithm 3: fixed M (adjust wt based on equilibrium dissolution) '''
      alphaM = self.origin_state[:,:,physics.get_state_slice("pDensityM")] \
        / rhoM
      alphaWv = 1.0 - alphaA - alphaM
      # Force positivity
      alphaWv[np.where(alphaWv < 0)] = 0
      alphaM = 1.0 - alphaA - alphaWv
      arhoWd = conc_eq / (1.0 + conc_eq) * alphaM * rhoM
      arhoWt = arhoWd + alphaWv * rhoWv
    else:
      raise NotImplementedError(f"Unknown constraint key: {constr_key}." + 
        "Implemented constraints are WtEq, WvEq, MEq (default).")

    # Compute volumetric energy
    e = alphaA * rhoA * physics.Gas[0]["c_v"] * T \
      + alphaWv * rhoWv * physics.Gas[1]["c_v"] * T \
      + alphaM * rhoM * (physics.Liquid["c_m"] * T + physics.Liquid["E_m0"])

    constrained_state[:,:,physics.get_state_slice("pDensityWv")] = alphaWv * rhoWv
    constrained_state[:,:,physics.get_state_slice("pDensityM")] = alphaM * rhoM
    constrained_state[:,:,physics.get_state_slice("Energy")] = e
    constrained_state[:,:,physics.get_state_slice("pDensityWt")] = arhoWt
   
    return constrained_state

  def set_initial_condition(self, p_bdry:float=1e5, p_bc_loc:str="x2",
    is_jump_included:bool=False, x_jump:float=0.0, is_x_jump_exact:bool=False,
    traction_fn:Callable=None, owner_domain=None):
    ''' Replace the initial condition in self.solver with hydrostatic
    condition.
    
    Inputs:
    p_bdry: pressure at boundary. If None, awaits and retrieves data from bnet
    p_bc_loc: location of pressure boundary condition
    is_jump_included: whether a pressure jump is included in the initial cond
    x_jump: nominal position for placing the pressure jump
    is_x_jump_exact: whether to force x_jump to be exact (True not recommended)
    traction_fn: function that specifies traction (momentum source term)
    '''

    solver = self.solver

    ''' Comms '''
    if p_bdry is None:
      if owner_domain is None:
        raise Exception("Boundary pressure value not provided, but owner "
          + "of domain in data network is unknown.")
      # Await boundary pressure 
      # Get id string of adjacent domain from domain graph (local)
      physics = self.solver.physics
      bkey = physics.BCs[p_bc_loc].bkey
      adjacent_domain_id = [
        key for key in 
        physics.domain_edges[physics.domain_id][bkey] 
        if key != physics.domain_id][0]
      # Get key for boundary state in shared memory (via Manager.dict)
      data_net_key = physics.edge_to_key(
                      physics.domain_edges[physics.domain_id][bkey],
                      adjacent_domain_id)
      # Return orientation-corrected exterior state
      Ub = physics.bdry_data_net[data_net_key]["bdry_face_state"]
      p_bdry = physics.compute_additional_variable("Pressure", Ub, True)[0,0,0]

    ''' Identify index and normal corresponding to bc loc '''
    filtered = [i for i in range(self.N) 
      if np.isclose(self.x[i], 
        solver.mesh.node_coords[
          solver.mesh.elem_to_node_IDs[
            solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].elem_ID]
          [solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].face_ID]])]
    if len(filtered) > 1:
      raise Exception("Multiple nodes at specified pressure BC location.")
    
    if len(filtered) >= 1:
      BC_index = filtered[0]
    else:
      # Take BC index corresponding to node closest to BC
      BC_index = np.argmin(np.abs(self.x - solver.mesh.node_coords[
          solver.mesh.elem_to_node_IDs[
            solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].elem_ID]
          [solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].face_ID]]))
    
    BC_normal_sign = solver.elem_helpers.normals_elems[
      solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].elem_ID,:,0,0][
        solver.mesh.boundary_groups[p_bc_loc].boundary_faces[0].face_ID]

    ''' Construct boundary load vector '''
    b = dok_array((self.N, 1,))
    b[BC_index] = BC_normal_sign * 0.5 * p_bdry
    b = b.tocsr()

    ''' Construct designated pressure jumps close to target position
    Algorithm: Map x -> phi(x) assuming 1D Lagrange segment by mapping from phys
    coordinate to reference space and evaluating phi(x) in reference space. '''
    
    delta_source = scipy.sparse.dok_array(b.shape)

    if is_jump_included:
      # Compute single pressure jump based on maximum pressure in unequilibrated
      # initial condition
      p_jump = -(np.max(self.solver.physics.compute_additional_variable(
        "Pressure", self.origin_state, True)) - p_bdry)
      self.x_jump_actual = x_jump

      if is_x_jump_exact:
        ''' Evaluate test function at target location.
        Delta distribution applied to test function evaluates the test function.
        The solution is locally discontinuous, and is thus not contained in the
        local polynomial space. It is recommended to not force the delta
        distribution to the desired location, but rather place it at an element
        boundary.
        '''
        # Find element intersecting x_jump
        indicated_indices =  [i for i in range(solver.mesh.num_elems)
          if x_jump > np.min(solver.mesh.elements[i].node_coords) 
          and x_jump <= np.max(solver.mesh.elements[i].node_coords)]
        # Select first element index to place Dirac mass
        target_elem_ID = indicated_indices[0]

        # Get Jacobian, assume const geometric Jacobian (affine transformation only)
        _, jac, _ = basis_tools.element_jacobian(solver.mesh, 0,
              np.array([0]), get_djac=False, get_jac=True, get_ijac=False)
        # Squeeze jacobian from [nq=1,ndims=1,ndims=1]
        J = jac[0,0,0]
        # Compute distance from min_x node in ref space
        ref_dist = (x_jump - np.min(solver.mesh.elements[target_elem_ID].node_coords))/J
        # Assume ref space is [-1, 1]
        ref_coord = -1.0 + ref_dist
        # Evaluate test functions at reference coordinate
        integrated_val = solver.basis.get_values(np.array(ref_coord,ndmin=2))
        # Add the test of source to interior of affected element
        delta_source[self.nb*target_elem_ID:self.nb*(target_elem_ID+1)] = p_jump*integrated_val
      else:
        '''Place delta mass at closest element boundary, splitting in half
        the mass to both neighbouring (1D) elements.'''
        # Find face closest to x_jump
        face = solver.mesh.interior_faces[
          np.argmin(np.abs(x_jump - solver.mesh.node_coords))]
        # Save x_jump snapped to closest face
        self.x_jump_actual = np.min(np.abs(x_jump - solver.mesh.node_coords))
        # Add pressure jump source
        # Adjust index by 1 due to INTERIOR face indexing (excludes bdry faces)
        target_idx = max(face.elemL_ID-1, face.elemR_ID-1)
        delta_source[self.nb*target_idx-1] = p_jump/2
        delta_source[self.nb*target_idx] = p_jump/2

        # Process origin state

    ''' Assemble local Lax-Friedrichs numerical flux matrix A '''
    A = dok_array((self.N, self.N,))
    nb = self.nb
    for i in range(self.ne):
          A[nb*i, nb*i] += -0.5
          A[nb*(i+1)-1, nb*(i+1)-1] += 0.5
          if i > 0:
              A[nb*i, nb*i-1] += -0.5
          if i < self.ne-1:
              A[nb*(i+1)-1, nb*(i+1)] += 0.5
    # Check for one-sided numerical fluxes where BC is absent
    if BC_index == 0:
      A[-1,-1] += 0.5
    elif BC_index == self.N-1:
      A[0,0] += -0.5
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
    M = dok_array((self.N, self.N,))
    nb = self.nb
    for i in range(self.ne):
      M[nb*i:nb*(i+1), nb*i:nb*(i+1)] = M_vec[i,:,:]
    M = M.tocsr()

    ''' Compute initial guess from average weight '''
    gsource = [s for s in solver.physics.source_terms if
      type(s) is GravitySource][0]
    # Barotropic gravity source function [ne, nq]
    g_fn = lambda p: gsource.get_source(
        self.solver.physics,
        self.eval_barotropic_state(p),
        self.solver.elem_helpers.x_elems,
        self.solver.time)[
          :,:,self.solver.physics.get_state_index("XMomentum")]
    vec_cast = lambda S: np.expand_dims(S.ravel(),axis=1)

    # Build source function
    S_fn = lambda p: g_fn(p)
    # Add traction
    if traction_fn is not None:
      S_fn = lambda p: g_fn(p) \
        + traction_fn(self.solver.elem_helpers.x_elems.squeeze(axis=2))
    
    # Define load vector function
    f_fn = lambda p: -b + M @ vec_cast(S_fn(p)) + delta_source

    ''' Compute initial guess '''
    rho0 = np.sum(
      solver.state_coeffs[:,:,solver.physics.get_mass_slice()],axis=2) \
      .ravel()[BC_index]
    p_like = 0.0*b.todense() + 1.0
    f = -b + (-rho0*gsource.gravity) * M @ p_like + delta_source
    p = scipy.sparse.linalg.spsolve(A-B, f)
    # Copy for internal debugging
    p_guess = p

    ''' Fixed point iteration '''
    fixedpointiter = lambda p: scipy.sparse.linalg.spsolve(A-B,
      f_fn(np.expand_dims(p,axis=1)))
    evalresidual = lambda p : np.linalg.norm(
      (A-B)@np.expand_dims(p,axis=1) - f_fn(np.expand_dims(p,axis=1)), 'fro')
    residuals = np.array([evalresidual(p)])
    # Start fixed point iteration
    for i in range(self.N_iter):
      p = fixedpointiter(p)
      fpi_res = evalresidual(p)
      residuals = np.append(residuals, fpi_res)
      if fpi_res < self.FPI_TOL:
        break

    # Save residual history
    self.residuals = residuals
    # Compute difference in pressure between pressure vector and eval'd state
    self.pError = np.linalg.norm(p - 
      self.solver.physics.compute_additional_variable("Pressure",
        self.eval_barotropic_state(p) ,True).ravel()
    )
    # Evaluate Quail-ready state
    U = self.eval_barotropic_state(p)
    # Check unsteady problem residual
    try:
      self.unsteady_residuals_face = np.zeros_like(U)
      self.solver.get_boundary_face_residuals(U, self.unsteady_residuals_face)
      self.solver.get_interior_face_residuals(U, self.unsteady_residuals_face)
      self.unsteady_residuals_elements = np.zeros_like(U)
      self.solver.get_element_residuals(U, self.unsteady_residuals_elements)
      self.unsteady_residuals = self.unsteady_residuals_face\
        + self.unsteady_residuals_elements
    except KeyError as err:
      # Data not found in boundary data net
      self.logger.info(f'''The following exception occurred due to evaluating
        residuals before boundary data was posted. If the residual is not needed
        for computation and is only used for verification, then this error can
        be safely ignored.''')
      self.logger.error(f"KeyError: {err}",stack_info=True)

    # plt.plot(self.x, p)

    # Replace solver state coefficients with hydrostatic state
    self.solver.state_coeffs = U

    # Replace output initial condition with equilibrated condition
    if self.solver.params["WriteInitialSolution"]:
      readwritedatafiles.write_data_file(self.solver, 0)
    