# ------------------------------------------------------------------------ #
#
#       File : src/multidomain.py
#
#       Defines a domain/subprocess abstraction that wraps solver.
#
# ------------------------------------------------------------------------ #

import numpy as np
import numerics.helpers.helpers as helpers
import logging

_global_timeout_seconds = 615.0

# TODO: typing of input args

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
    ''' Start synchronized solve, injecting to self.solver.solve via the
     customuserfunction API.
    '''
    # Check validity of object
    if None in [self.glight_condition, self.ready_queue,
                self.bdry_data_net, self.domain_output_dict]:
      raise Exception('Initialization of Domain object incomplete.')
    # Set post-await as user function in solve API
    self.solver.custom_user_function = self.post_await
    # Call solver solve
    self.solver.solve()
    # Send output to main process
    self.domain_output_dict[self.id] = self.solver
  
  def pprint(self, msg):
    if self.verbose:
      print(f"<Process {self.id}>: {msg}")

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
      # Get data from BC method
      data["bdry_face_state"] = BC.get_extrapolated_state(solver.physics, UqI, normals, x, None)
      ''' Attach data for elements at boundary to payload. '''
      # Identify elements at boundary
      boundary_elem_IDs = [bface.elem_ID for bface 
                           in solver.mesh.boundary_groups[localbname].boundary_faces]
      data["element_data"] = solver.state_coeffs[boundary_elem_IDs,:,:]

      ''' Post data to network '''
      write_key = Domain.edge_to_key(self.domain_edges[self.id][bname], self.id)
      self.pprint(f"Write to key <{write_key}>")
      self.bdry_data_net[write_key] = data

    # Await green light
    with self.glight_condition:
        self.ready_queue.put(self.id)
        is_success = self.glight_condition.wait(timeout=Domain.wait_timeout)
        if not is_success:
          raise Exception("Waiting for green-light timed out.")

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