# ------------------------------------------------------------------------ #
#
#       File : src/multidomain.py
#
#       Defines a domain/subprocess abstraction that wraps solver.
#
# ------------------------------------------------------------------------ #

import numpy as np
import numerics.helpers.helpers as helpers
from dataclasses import dataclass
from typing import Callable, Union
import logging
import queue
import time


_global_timeout_seconds = 10*3600.0

# TODO: typing of input args
# TODO: parallel setup of domains

class Domain():
  ''' Subprocess wrapper for a solver object. '''
  wait_timeout = _global_timeout_seconds

  def __init__(self, solver, physics, mesh, id, verbose=False):
    self.solver = solver
    self.physics = physics
    self.mesh = mesh
    self.id = id
    self.domain_edges = {}
    self.edge_name_to_loc_bdry = {}
    self.num_timesteps = np.nan
    self.glight_condition = None
    self.ready_queue = None
    self.bdry_data_net = None
    self.domain_output = None
    self.verbose = verbose
    self.user_functions = Domain.UserFunctionSeq()
    self.total_await_time = 0.0
    self.begin_time = time.time()
    self.total_run_time = 0.0

    # Initalize logger
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.DEBUG)
    h = logging.FileHandler(
      filename=f"multidomain_id{id}_{hash(solver)}.log",
      encoding="utf-8")
    h.setFormatter(logging.Formatter(
      '[%(asctime)s][%(levelname)s] Logger <%(name)s> : %(message)s'))
    self.logger.addHandler(h)
  
  def set_network(self, glight_condition, ready_queue,
                  bdry_data_net, domain_output_dict):
    self.glight_condition = glight_condition
    self.ready_queue = ready_queue
    self.bdry_data_net = bdry_data_net
    self.domain_output_dict = domain_output_dict
    # Add tag data for data consumers in physics
    self.physics.bdry_data_net = self.bdry_data_net
    self.physics.domain_id = self.id
    self.physics.domain_edges = self.domain_edges
    self.physics.edge_to_key = Domain.edge_to_key
  
  @staticmethod
  def edge_to_key(edge_tuple, node_id):
    ''' Converts edge tuple (node1:str, node2:str) and node_id:str to str. ''' 
    edge_sorted = sorted(list(edge_tuple), key=str.lower)
    return f"{edge_sorted[0]}-{edge_sorted[1]}-{node_id}"

  def solve(self):
    ''' Start synchronized solve, adding routines to self.solver.solve via
        Quail's custom_user_function API.
    '''
    
    ''' Check validity of this object. '''
    if None in [self.glight_condition, self.ready_queue,
                self.bdry_data_net, self.domain_output_dict]:
      raise Exception('Initialization of Domain object incomplete.')
    
    ''' Custom user function injection. '''
    # Get Quail-style custom user function loaded by input file
    if self.solver.custom_user_function is not None:
      # Append function to sequence of functions as a single callable
      self.user_functions.append_always(self.solver.custom_user_function)
    # Set post-await as user function as last function
    self.user_functions.append_always(self.post_await)

    # Provide sequence of custom users functions to Quail API
    self.solver.custom_user_function = self.user_functions

    ''' Additional warmup procedures '''
    # Get source terms by name
    source_term_names = [source.__class__.__name__.casefold()
                         for source in self.solver.physics.source_terms] 
    # Check for sources that depend on solution divergence
    if "FragmentationStrainRateSource".casefold() in source_term_names:
      # Assign solver directly to source term
      self.solver.physics.source_terms[source_term_names.index("FragmentationStrainRateSource".casefold())].solver = self.solver

    ''' Run solver.solve'''
    # Call solver solve
    self.solver.solve()
    # Send output to main process
    self.domain_output_dict[self.id] = self.solver
    # End timer
    self.total_run_time = time.time() - self.begin_time
    # Post data to bdry_data_net (assume current object will no longer be valid
    # on the main process)
    self.bdry_data_net[self.id] = {
      "total_run_time": self.total_run_time,
      "total_await_time": self.total_await_time,
    }
  
  def pprint(self, msg):
    if self.verbose:
      print(f"<Process {self.id}>: {msg}")

  class UserFunctionSeq(Callable):
    '''  Sequence of functions with call order as specified. 

    Functions are wrapped in callables of type UserFunctionSeq.CallbackType
    that allow user to specify whether the function should be called the first
    time UserFunctionSeq is called, and whether the function should be called
    every subsequent time.

    Returns of functions are discarded.
    '''

    def __init__(self):
      self.callbacks = []
      self.call_count = 0
    
    class CallbackType():
      ''' Wrapper for callback with state.
        
        function: the function to wrap
        initial: flags whether callback is designated for 1st call to user funcs
        postinitial: flag whether callback is designated for > 1st call
      '''
      def __init__(self, function:Callable,
          initial:bool=True, postinitial:bool=True, owner_domain=None):
        self.function = function
        self.initial = initial
        self.postinitial = postinitial
        self.owner_domain = owner_domain

      def __call__(self, solver):
        if self.owner_domain is None:
          return self.function(solver)
        else:
          return self.function(solver, owner_domain=self.owner_domain)
      

    def __call__(self, solver):
      ''' Calls each function in sequence, accounting for initials and
      postinitials. Supports funtions with positional arg (solver).'''
      for callback in self.callbacks:
        if self.call_count == 0 and callback.initial or \
           self.call_count >= 1 and callback.postinitial:
          callback(solver)
      self.call_count += 1

    def append(self, functions,
      initial:bool=True, postinitial:bool=True, owner_domain=None):
      ''' Add function to function sequence, specifying whether active for
      initial call and post initial call.'''

      try:
        for value in functions:
          self.callbacks.append(
            __class__.CallbackType(value, initial, postinitial, owner_domain))
      except TypeError:
        # `functions` as a single Callable
        self.callbacks.append(
            __class__.CallbackType(functions, initial, postinitial, owner_domain))
 
    def append_always(self, functions):
      ''' Add function to function sequence, and have it execute every time
      sequence is called.'''
      self.append(functions, initial=True, postinitial=True)

    def append_initial(self, functions):
      ''' Add function to function sequence, and have it execute only the first
      time the sequence is called. '''
      self.append(functions, initial=True, postinitial=False)

    def append_noninitial(self, functions):
      ''' Add function to function sequence, and have it execute except the
      first time the sequence is called. '''
      self.append(functions, initial=False, postinitial=True)

  def post_await(self, solver):
    ''' Post data and await green light signal from observer node.'''
    
    # Test boundary at x2
    # self.domain_edges[self.id]={"x2":(self.id, "virtualDomain")}
    data = {}
    # Post data
    for bname in self.domain_edges[self.id]:
      # Get boundary name known to solver.mesh from the multidomain edge name
      localbname = self.edge_name_to_loc_bdry[self.id][bname]
      ''' Call BC class's boundary data computation method. '''
      bgroup = solver.mesh.boundary_groups[localbname]
      basis_val = solver.bface_helpers.faces_to_basis[
        solver.bface_helpers.face_IDs[bgroup.number]] # [nbf, nq, nb]
      # Interpolate state at quad points
      UqI = helpers.evaluate_state(
        solver.state_coeffs[solver.bface_helpers.elem_IDs[bgroup.number]],
        basis_val) # [nbf, nq, ns]
      # Get boundary geom data
      normals = solver.bface_helpers.normals_bgroups[bgroup.number] # [nbf, nq, ndims]
      x = solver.bface_helpers.x_bgroups[bgroup.number] # [nbf, nq, ndims]
        # quad_wts = solver.bface_helpers.quad_wts
      BC = solver.physics.BCs[bgroup.name]

      # Compute ordinary boundary state
      bdry_state = BC.get_extrapolated_state(solver.physics, UqI, normals, x, None)
      # Compute averaged data
      if solver.physics.NDIMS == 2:
        # Compute boundary measure
        boundary_length = solver.bface_helpers.face_lengths_bgroups[bgroup.number].sum()
        # Check geometry by presence of geometric source term
        if True in ["CylindricalGeometricSource" in str(source_term.__class__)
            for source_term in solver.physics.source_terms]:
          # Compute state averaged over the boundary (Polar)
          # e.g. for interval [0, a], boundary_measure is 0.5 * a**2
          boundary_measure = np.einsum('ijm, jn, ij -> ',
            np.linalg.norm(normals, axis=2, keepdims=True),
            solver.bface_helpers.quad_wts,
            x[:,:,0] # r coordinate
            )
          data["bdry_face_state_averaged"] = np.expand_dims(
            np.einsum('ijm, ijk, jn, ij -> k',
            np.linalg.norm(normals, axis=2, keepdims=True),
            UqI,
            solver.bface_helpers.quad_wts,
            x[:,:,0] # r coordinate
            ) / boundary_measure, 
            axis=(0,1))
        else:
          # Compute state averaged over the boundary (Cartesian)
          data["bdry_face_state_averaged"] = np.expand_dims(
            np.einsum('ijm, ijk, jn -> k',
            np.linalg.norm(normals, axis=2, keepdims=True),
            UqI,
            solver.bface_helpers.quad_wts) / boundary_length,
            axis=(0,1))

      # Get data from BC method
      data["bdry_face_state"] = bdry_state
      ''' Attach data for elements at boundary to payload. '''
      # # Identify elements at boundary
      # boundary_elem_IDs = [bface.elem_ID for bface 
      #                      in solver.mesh.boundary_groups[localbname].boundary_faces]
      # data["element_data"] = solver.state_coeffs[boundary_elem_IDs,:,:]

      ''' Post data to network '''
      write_key = Domain.edge_to_key(self.domain_edges[self.id][bname], self.id)
      self.pprint(f"Write to key <{write_key}>")
      self.bdry_data_net[write_key] = data

    # Await green light
    await_time = time.time()
    with self.glight_condition:
        self.ready_queue.put(self.id)
        try:
          is_success = self.glight_condition.wait(timeout=Domain.wait_timeout)
          if not is_success:
            raise Exception("Waiting for green-light timed out.")
        except queue.Empty as err:
          self.logger.info(f'''The following exception occurred due to
          multidomain.Domain timing out while waiting for the green light
          condition from boundary data network. If this occured while the
          code is paused in a debugger, this error can be ignored.''')
          self.logger.error(f"KeyError: {err}",stack_info=True)
    self.total_await_time += time.time() - await_time

class Observer():
  ''' Observer that oversees synchronization of multiple domains. '''
  get_timeout = _global_timeout_seconds

  def __init__(self, num_timesteps, ready_queue, glight_condition, dom_count):
    self.num_timesteps = num_timesteps
    self.ready_queue = ready_queue
    self.glight_condition = glight_condition
    self.dom_count = dom_count

  def listen(self):
    # Initialize counter of # domains ready
    num_ready = 0
    # Global timestep counter
    completed_timesteps = 0
    if not self.num_timesteps > 0:
      raise Exception("Expecting to run for zero timesteps.")
    # Perform N+1 signal cycles (since custom function is called N+1 times)
    while completed_timesteps <= self.num_timesteps:
      # Count ready-messages in queue
      msg = self.ready_queue.get(block=True, timeout=Observer.get_timeout)
      num_ready += 1
      # Send green-light signal when one signal received per domain
      if num_ready == self.dom_count:
        with self.glight_condition:
          self.glight_condition.notify_all()
          # Reset counters
          num_ready = 0
          completed_timesteps += 1
