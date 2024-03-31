import numpy as np

import numerics.helpers.helpers as helpers
import solver.tools as solver_tools
import physics.multiphasevpT.atomics as atomics

def velocity(physics, U:np.array, ndims:int) -> np.array:
  ''' Computes vector field u to take a divergence of. '''
  rho = U[...,0:3].sum(axis=-1, keepdims=True)
  # Return velocity field with u, v in the flux axis
  return np.expand_dims(U[...,3:3+ndims] / rho, axis=-2)

def pressure_x_vector(physics, U:np.array, ndims:int) -> np.array:
  ''' Computes vector field (p,0) to take a divergence of. '''
  # Compute pressure using compute atomics
  T = atomics.temperature(U[...,0:3], U[...,3:3+ndims], U[...,3+ndims:4+ndims], physics)
  phi = atomics.gas_volfrac(U[...,0:3], T, physics)
  p = atomics.pressure(U[...,0:3], T, phi, physics)
  if ndims == 1:
    return p[...,np.newaxis]
  # Stack flux direction in new axis -1
  return np.stack((p, np.zeros_like(p)), axis=-1)

def pressure_y_vector(physics, U:np.array, ndims:int) -> np.array:
  ''' Computes vector field (0,p) to take a divergence of.
  Returns a zero array if ndims==1. '''
  if ndims == 1:
    return np.zeros_like(U[...,0:1,np.newaxis])
  # Compute pressure using compute atomics
  T = atomics.temperature(U[...,0:3], U[...,3:3+ndims], U[...,3+ndims:4+ndims], physics)
  phi = atomics.gas_volfrac(U[...,0:3], T, physics)
  p = atomics.pressure(U[...,0:3], T, phi, physics)
  # Stack flux direction in new axis -1
  return np.stack((np.zeros_like(p), p), axis=-1)

def work_flux(physics, U:np.array, ndims:int) -> np.array:
  ''' Computes vector field pu to take a divergence of. '''
  # Compute pressure using compute atomics
  T = atomics.temperature(U[...,0:3], U[...,3:3+ndims], U[...,3+ndims:4+ndims], physics)
  phi = atomics.gas_volfrac(U[...,0:3], T, physics)
  p = atomics.pressure(U[...,0:3], T, phi, physics)
  # Stack flux direction in new axis -1
  rho = U[...,0:3].sum(axis=-1, keepdims=True)
  # Return velocity field with u, v in the flux axis
  return np.expand_dims(U[...,3:3+ndims] / rho * p, axis=-2)

def Upwind2D(vector_field_U:callable, UqL:np.array, UqR:np.array,
             normals:np.array) -> np.array:
  ''' Upwind flux in 2D, using sign of velocity. '''
  ndims = normals.shape[-1]
  # Compute flux traces
  FqL = np.einsum('ijkl, ijl -> ijk', vector_field_U(UqL), normals)
  FqR = np.einsum('ijkl, ijl -> ijk', vector_field_U(UqR), normals)
  # Upwinding by arithmetic average
  return np.where(FqL + FqR >= 0, FqL, FqR)

def Central2D(vector_field_U:callable, UqL:np.array, UqR:np.array,
              normals:np.array) -> np.array:
  ''' Central flux in 2D. '''
  ndims = normals.shape[-1]
  # Compute flux traces
  FqL = np.einsum('ijkl, ijl -> ijk', vector_field_U(UqL), normals)
  FqR = np.einsum('ijkl, ijl -> ijk', vector_field_U(UqR), normals)
  # Upwinding by arithmetic average
  return 0.5 * (FqL + FqR)

def eval_strainrate(solver, U:np.array, x, t) -> np.array:
  ''' Compute strain rate as proportional to material derivative of pressure.
  '''

  # Evaluat
  Uq = helpers.evaluate_state(U,
                              solver.elem_helpers.basis_val,
                              skip_interp=solver.basis.skip_interp)  

  # x = x_elem_quad
  physics = solver.physics

  # Compute source contribution
  sources = [source.get_source(physics, Uq, x, t)
                          for source in physics.source_terms]
  if len(sources) == 0:
    sources = [np.zeros_like(x)]
  source_sum = np.stack(sources, axis=-1).sum(axis=-1)

  _, divu_quad = eval_divu(solver, velocity, U)
  _, divpu_quad = eval_divu(solver, work_flux, U)
  _, dpdx_quad = eval_divu(solver, pressure_x_vector, U)
  _, dpdy_quad = eval_divu(solver, pressure_y_vector, U)

  # Assemble non-advective terms for material derivative computation
  non_advective_terms = np.zeros(Uq.shape)
  non_advective_terms[...,3:4] = dpdx_quad
  non_advective_terms[...,4:5] = dpdy_quad
  non_advective_terms[...,5:6] = divpu_quad
  # Compute material derivative at quadrature points
  DUDt = source_sum - Uq * divu_quad - non_advective_terms
  # Compute pressure state-gradient
  p_sgrad = solver.physics.compute_pressure_sgradient(Uq)
  # Compute material derivative of pressure through chain rule
  DpDt = np.einsum("ijk, ijk -> ij", p_sgrad, DUDt)[:,:,np.newaxis]

  # Compute proportional volumetric dilation
  rho0, K = solver.physics.Liquid["rho0"], solver.physics.Liquid["K"]
  rho = Uq[...,0:3].sum(axis=-1, keepdims=True)
  # Return magma strain rate
  return (-rho0 * rho0 / K) * DpDt / (rho*rho)

def eval_divu(solver, vector_field:callable, U:np.array) -> tuple:
  ''' Compute divergence of velocity at quadrature points.
  Modeled after the residual computation in Quail without diffusive terms.
  Computes div u in the DG weak form in the following steps.
    1. Boundary face contributions, using the trace as the boundary value.
    2. Interior face contributions, explicitly upwinding based on {{u}}.
    3. Volume contributions
  Finally, multiplying by the inverse mass matrix gives div u at nodes.
  
  Returns div u both at degrees of freedom and mapped to quadrature points.
  '''

  # Select numerical flux
  num_flux = Upwind2D
  # Curry vector field to function of U
  vector_field_U = lambda U: vector_field(solver.physics,
                                          U,
                                          solver.physics.NDIMS)
  mesh = solver.mesh
  bface_helpers = solver.bface_helpers
  elem_helpers = solver.elem_helpers
  int_face_helpers = solver.int_face_helpers
  
  # Allocate divergence of vector field
  divu = np.zeros((*U.shape[:-1], 1))

  ''' Boundary faces '''
  for bgroup in mesh.boundary_groups.values():
    # Identify faces included in bgroup
    bgroup_faces = bface_helpers.face_IDs[bgroup.number]
    # Evaluate basis functions at boundary face quad points [nbf, nq, nb]
    basis_val = solver.bface_helpers.faces_to_basis[bgroup_faces]
    # Interpolate state at quad points [nbf, nq, ns]
    UqI = helpers.evaluate_state(U[bface_helpers.elem_IDs[bgroup.number]],
                                 basis_val)  
    # Use trace as boundary value
    Fq = np.einsum("ijkl, ijl -> ijk",
                    vector_field_U(UqI),
                    bface_helpers.normals_bgroups[bgroup.number])
    # Compute contribution to adjacent element residual [nf, nb, ns]
    _contribution = np.einsum('ijn, jm, ijk -> ink',
                              basis_val,
                              bface_helpers.quad_wts,
                              Fq)
    # Sum contribution for div u
    np.add.at(divu, bface_helpers.elem_IDs[bgroup.number], _contribution)

  ''' Interior faces '''    
  # Select bases that correspond to the L and R traces of faces
  basesL = int_face_helpers.faces_to_basisL[int_face_helpers.faceL_IDs]
  basesR = int_face_helpers.faces_to_basisR[int_face_helpers.faceR_IDs]
  # Compute traces at interior faces [nf, nq, ns]
  UqL = helpers.evaluate_state(U[int_face_helpers.elemL_IDs], basesL)
  UqR = helpers.evaluate_state(U[int_face_helpers.elemR_IDs], basesR)
  # Compute numerical flux [nf, nq, ns] replacing physics.get_conv_flux_numerical
  Fq = num_flux(vector_field_U, UqL, UqR, int_face_helpers.normals_int_faces)
  # Compute contribution to left and right element residuals
  #   (solver_tools.calculate_boundary_flux_integral)
  resL = np.einsum('ijn, jm, ijk -> ink',
                    basesL,
                    int_face_helpers.quad_wts,
                    Fq)
  resR = np.einsum('ijn, jm, ijk -> ink',
                    basesR,
                    int_face_helpers.quad_wts,
                    Fq)
  
  # Add this residual back to the global. The np.add.at function is
  # used to correctly handle duplicate element IDs.
  np.add.at(divu, int_face_helpers.elemL_IDs, resL)
  np.add.at(divu, int_face_helpers.elemR_IDs, -resR)    

  ''' Element residual '''
  # Interpolate state at volumetric quad points [ne, nq, ns]
  Uq = helpers.evaluate_state(U, elem_helpers.basis_val,
      skip_interp=solver.basis.skip_interp) 
  # Evaluate the volumetric contribution to div u
  #   (replaces solver_tools.calculate_volume_flux_integral
  #   and physics.get_conv_flux_interior(Uq)[0])
  divu -= np.einsum('ijnl, jm, ijm, ijkl -> ink',
                    elem_helpers.basis_phys_grad_elems,
                    elem_helpers.quad_wts,
                    elem_helpers.djac_elems,
                    vector_field_U(Uq))
  
  # Invert mass matrix to get div u at node points
  divu = np.einsum('ijk, ikl -> ijl', elem_helpers.iMM_elems, divu)

  # Map div u to quadrature points
  divu_quad = np.einsum('jn, ink -> ijk', elem_helpers.basis_val, divu)

  return divu, divu_quad

def _eval_divu_additional(solver, U:np.array) -> np.array:
    ''' Compute divergence of velocity at quadrature points.
    Modeled after the residual computation in Quail without diffusive terms.
     
    Commented out some diffusive flux parts. 
    '''
    mesh = solver.mesh
    physics = solver.physics
    bface_helpers = solver.bface_helpers
    elem_helpers = solver.elem_helpers
    int_face_helpers = solver.int_face_helpers
    
    # Allocate divergence of vector field
    divu = np.zeros((*U.shape[:-1], 1))    

    ''' Boundary faces '''

    for bgroup in mesh.boundary_groups.values():
      # Identify faces included in bgroup
      bgroup_faces = bface_helpers.face_IDs[bgroup.number]
      # Evaluate basis functions at boundary face quad points [nbf, nq, nb]
      basis_val = solver.bface_helpers.faces_to_basis[bgroup_faces]
      # Interpolate state at quad points [nbf, nq, ns]
      UqI = helpers.evaluate_state(U[bface_helpers.elem_IDs[bgroup.number]],
                                   basis_val)  

      # Evaluate gradient of state at quad points
      # basis_ref_grad = solver.bface_helpers.faces_to_basis_ref_grad[bgroup_faces] 
      # gUq_ref = solver.evaluate_gradient(U[bface_helpers.elem_IDs[bgroup.number]],
      #                                    basis_ref_grad)
      # gUq = solver.ref_to_phys_grad(bface_helpers.ijac_bgroups[bgroup.number],
      #                               gUq_ref)
      # Compute boundary flux
      # Fq, FqB = physics.BCs[bgroup.name].get_boundary_flux(
      #    physics,
      #    UqI,
      #    bface_helpers.normals_bgroups[bgroup.number],
      #    bface_helpers.x_bgroups[bgroup.number],
      #    solver.time,
      #    gUq=gUq)
      # Use trace as boundary value
      Fq = vector_field(UqI, solver.physics.NDIMS)

      # Compute contribution to adjacent element residual
      _contribution = solver_tools.calculate_boundary_flux_integral(
          basis_val, bface_helpers.quad_wts, Fq)
      # Diffusive contribution
      # FqB_phys = solver.ref_to_phys_grad(bface_helpers.ijac_bgroups[bgroup.number], FqB)
      # _diff_contribution = solver.calculate_boundary_flux_integral_sum(
      #         basis_ref_grad, quad_wts, FqB_phys)

      np.add.at(divu, bface_helpers.elem_IDs[bgroup.number], -_contribution)

    ''' Interior faces '''
    
    # Select bases that correspond to the L and R traces of faces
    basesL = int_face_helpers.faces_to_basisL[int_face_helpers.faceL_IDs]
    basesR = int_face_helpers.faces_to_basisR[int_face_helpers.faceR_IDs]

    # Compute traces at interior faces [nf, nq, ns]
    UqL = helpers.evaluate_state(U[int_face_helpers.elemL_IDs], basesL)
    UqR = helpers.evaluate_state(U[int_face_helpers.elemR_IDs], basesR)

    # Interpolate gradient of state at quad points
    # gUqL_ref = solver.evaluate_gradient(UL, 
    #     int_face_helpers.faces_to_basis_ref_gradL[faceL_IDs])
    # gUqR_ref = solver.evaluate_gradient(UR, 
    #     int_face_helpers.faces_to_basis_ref_gradR[faceR_IDs])
    # gUqL = solver.ref_to_phys_grad(int_face_helpers.ijacL_elems, gUqL_ref)
    # gUqR = solver.ref_to_phys_grad(int_face_helpers.ijacR_elems, gUqR_ref)

    # Compute numerical flux [nf, nq, ns] replacing physics.get_conv_flux_numerical
    Fq = num_flux(UqL, UqR, int_face_helpers.normals_int_faces)
    # Compute contribution to left and right element residuals
    resL = solver_tools.calculate_boundary_flux_integral(
        basesL, int_face_helpers.quad_wts, Fq)
    resR = solver_tools.calculate_boundary_flux_integral(
       basesR, int_face_helpers.quad_wts, Fq)
    
    # Add this residual back to the global. The np.add.at function is
    # used to correctly handle duplicate element IDs.
    np.add.at(divu, int_face_helpers.elemL_IDs, -resL)
    np.add.at(divu, int_face_helpers.elemR_IDs,  resR)    

    ''' Element residual '''
    # Interpolate state at volumetric quad points [ne, nq, ns]
    Uq = helpers.evaluate_state(U, elem_helpers.basis_val,
        skip_interp=solver.basis.skip_interp) 
    Fq = vector_field(Uq, solver.physics.NDIMS)
    # Evaluate the inviscid flux integral
    # Fq = physics.get_conv_flux_interior(Uq)[0] # [ne, nq, ns, ndims]
    divu += solver_tools.calculate_volume_flux_integral(
        solver, elem_helpers, Fq) # [ne, nb, ns]

    return divu