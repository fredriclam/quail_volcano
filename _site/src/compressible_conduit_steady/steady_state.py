''' Computes steady state. In the steady-state, the mass equation simplifies
to a constraint on constant mass flux j = j0. The momentum equation in terms
or mixture pressure, the energy equation in terms of enthalpy, and the mass
exchange between phases are modeled in terms of three coupled ODEs.
Mass fractions of non-reacting phases are constant in the steady-state as a
consequence of the mass equation.

Sample usage:

import matplotlib.pyplot as plt
import steady_state as ss
import numpy as np
f = ss.SteadyState(1e5, 1,
  override_properties={
    "yC": 0.00
  })
x = np.linspace(-4150, -150, 1000) 
U = f(x)
for i in range (8): 
  plt.subplot(3,3,i+1)
  plt.plot(x, U[...,i])
plt.show()

'''

import numpy as np
import scipy
import scipy.integrate
from scipy.special import erf
import matplotlib.pyplot as plt

try:
  import material_properties as matprops
except ModuleNotFoundError:
  import compressible_conduit_steady.material_properties as matprops

# DONE: TEST: why is the limit tau_d -> Infty giving negative exsolved mass?
#  -- too much pressure loss before it reaches the top. Events p->0, h->0 added.

# TODO: print->log
# TODO: plot interactive checks
# TODO in material_properties: interactive checks, thermopotential surface plotting


class SteadyState():

  # Global caching: if checkhash and pressure match, re-use value
  checkhash = 0
  last_crit_inputs = np.array([0, 0, 0, 0])
  cached_j0_p = (None, None)

  def __init__(self, x_global:np.array, p_vent:float, inlet_input_val:float,
    input_type:str="u", override_properties:dict=None,
    use_static_cache:bool=False, skip_rootfinding=False):
    ''' 
    Steady state ODE solver.
    Inputs:
      x_global: global x coordinates (necessary to coordinate solution between
        several 1D patches)
      p_vent: vent pressure (if small enough, expect flow choking).
      inlet_input_val: chamber inlet value of velocity u, pressure p, or mass
        flux j. Provide the corresponding type in input_type.
      input_type: "u", "p", or "j" as provided by user.
      override_properties (dict): map from property name to value. See first
        section of __init__ for overridable properties.
      use_static_cache: If use_static_cache is True, inlet condition
        determination is not done again if the solution was already computed
        by any instance of SteadyState and the existing checkhash and
        critical inputs match the saved run.
    Call this object to sample the solution at a grid x, consistent with the
    provided value of conduit_length.
    Providing the number of elements at initialization helps, since no
    re-evaluation of the numerical solution is required. Re-evaluation risks
    perturbing the location of a sonic boundary. A one-node correction is
    included against this case, extrapolating the value at the vent. See
    __call__ for this implementation.
    '''
    # Validate properties
    if override_properties is None:
      override_properties = {}
    self.override_properties = override_properties.copy()
    ''' Set default and overridable properties'''
    # Water mass fraction (uniformly applied in conduit)
    self.yWt            = self.override_properties.pop("yWt", 0.03)
    self.yA             = self.override_properties.pop("yA", 1e-7)
    # Water vapour presence slightly above zero for numerics in unsteady case
    #   Higher makes numerics more happy, even for steady-state exsolution
    #   1e-5 is 10 ppm
    self.yWvInletMin    = self.override_properties.pop("yWvInletMin", 1e-5)
    # Crystal content (mass fraction; must be less than yl = 1.0 - ywt)
    self.yC             = self.override_properties.pop("yC", 1e-7)
    self.yCMin          = self.override_properties.pop("yCMin", 1e-7)
    # Inlet fragmented mass fraction
    self.yFInlet        = self.override_properties.pop("yFInlet", 0.0)
    # Critical volume fraction
    self.crit_volfrac   = self.override_properties.pop("crit_volfrac", 0.7)
    # Exsolution timescale
    self.tau_d          = self.override_properties.pop("tau_d", 1.0)
    # Fragmentation timescale
    self.tau_f          = self.override_properties.pop("tau_f", 1.0)
    # Viscosity (Pa s)
    self.mu             = self.override_properties.pop("mu", 1e5)
    # Conduit dimensions (m)
    self.conduit_radius = self.override_properties.pop("conduit_radius", 50)
    # Chamber conditions
    self.T_chamber      = self.override_properties.pop("T_chamber", 800+273.15)
    # Gas properties
    self.c_v_magma      = self.override_properties.pop("c_v_magma", 3e3)
    self.rho0_magma     = self.override_properties.pop("rho0_magma", 2.7e3)
    self.K_magma        = self.override_properties.pop("K_magma", 10e9)
    self.p0_magma       = self.override_properties.pop("p0_magma", 5e6)
    self.solubility_k   = self.override_properties.pop("solubility_k", 5e-6)
    self.solubility_n   = self.override_properties.pop("solubility_n", 0.5)
    # Whether to neglect deformation energy in magma EOS
    self.neglect_edfm   = self.override_properties.pop("neglect_edfm", False)
    self.fragsmooth_scale = self.override_properties.pop("fragsmooth_scale", 0.01)
    # Strain rate criterion
    self.crit_strain_rate_k = self.override_properties.pop("crit_strain_rate_k", 0.01)
    # self.crit_strain_rate = self.crit_strain_rate_k * self.K_magma / self.mu
    # Fragmentation criterion selection
    self.fragmentation_criterion = self.override_properties.pop("fragmentation_criterion", "VolumeFraction")

    # Bind fragmentation source
    if self.fragmentation_criterion.casefold() == "VolumeFraction".casefold():
      self.frag_source = self.frag_source_volfrac
    elif self.fragmentation_criterion.casefold() == "StrainRate".casefold():
      self.frag_source = self.frag_source_strain_rate
    else:
      raise ValueError(f"Unknown fragmentation criterion with string " +
                       f"{self.fragmentation_criterion}. Try " +
                       f"VolumeFraction, or StrainRate.")

    # Debug option (caches AinvRHS, source term F as lambdas that cannot be
    # pickled by the default pickle module)
    self._DEBUG = False

    self.use_static_cache = use_static_cache

    # Output mesh
    self.x_mesh = x_global.copy()
    # Internal computation mesh
    self.x_mesh_native = x_global.copy()

    self.conduit_length = x_global.max() - x_global.min()
    # Compute liquid melt fraction
    self.yL = 1.0 - (self.yC + self.yA + self.yWt)

    # Input validation
    if self.yC + self.yA + self.yWt > 1:
      raise ValueError(f"Component mass fractions are [" +
        f"{self.yC, self.yA, self.yWt, self.yL}] for " +
        f"[crystal, air, water, melt].")
    if self.yFInlet > self.yL:
      raise ValueError(f"Inlet fragmented mass fraction exceeds the liquid" +
      f"melt mass fraction: inlet fragmented {self.yFInlet}, liquid melt {self.yL}.")
    if len(self.override_properties.items()) > 0:
      raise ValueError(
        f"Unused override properties:{list(self.override_properties.keys())}")

    # Set depth of conduit inlet
    self.x0 = x_global.min()

    # Set mixture properties
    mixture = matprops.MixtureMeltCrystalWaterAir()
    mixture.magma = matprops.MagmaLinearizedDensity(c_v=self.c_v_magma,
      rho0=self.rho0_magma, K=self.K_magma,
      p_ref=self.p0_magma, neglect_edfm=self.neglect_edfm)
    mixture.k, mixture.n = self.solubility_k, self.solubility_n
    self.mixture = mixture

    # Set tolerance for numerical solve
    self.brent_atol = 1e-5

    # Check static cache using hash of unpopped dict override_properties
    propshash = hash(tuple(override_properties.items()))
    inputs_array = np.array([x_global.min(), x_global.max(), 
      p_vent, inlet_input_val])
    if skip_rootfinding:
      pass
    elif use_static_cache \
      and propshash == SteadyState.checkhash \
      and np.all(inputs_array == SteadyState.last_crit_inputs):
      self.j0, self.p_chamber = SteadyState.cached_j0_p
    else:
      SteadyState.checkhash = propshash
      SteadyState.last_crit_inputs = inputs_array.copy()
      # Run once at the provided values
      self._set_cache(p_vent, inlet_input_val, input_type=input_type)
      SteadyState.cached_j0_p = (self.j0, self.p_chamber)

    # RHS function cache
    if self._DEBUG:
      self.F = None

  ''' Define partially evaluated thermo functions in terms of (p, h, y)'''

  def T_ph(self, p, h, y):
    return self.mixture.T_ph(p, h, self.yA, y, 1.0-y-self.yA)

  def v_mix(self, p, T, y):
    return self.mixture.v_mix(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dp(self, p, T, y):
    return self.mixture.dv_dp(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dh(self, p, T, y):
    return self.mixture.dv_dh(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dy(self, p, T, y):
    return self.mixture.dv_dy(p, T, self.yA, y, 1.0-y-self.yA)

  def vf_g (self, p, T, y):
    return self.mixture.vf_g(p, T, self.yA, y, 1.0-y-self.yA)

  def x_sat(self, p):
    return self.mixture.x_sat(p)

  def y_wv_eq(self, p, yWt=None):
    if yWt is None:
      yWt = self.yWt
    return self.mixture.y_wv_eq(p, yWt, self.yC)

  def A(self, p, h, y, yf, j0):
    ''' Return coefficient matrix to ODE (A in A dq/dx = f(q)).
    Coefficient matrix has block structure:
    [_ _ _ 0] [p ]
    [_ _ 0 0] [h ]
    [0 0 _ 0] [y ]
    [0 0 0 _] [yf]
     '''
    A = np.zeros((4,4))
    # Evaluate mixture state
    T = self.T_ph(p, h, y)
    v = self.v_mix(p, T, y)
    # Construct coefficient matrix
    A[0,:] = [j0**2 * self.dv_dp(p,T,y) + 1.0,
              j0**2 * self.dv_dh(p,T,y),
              j0**2 * self.dv_dy(p,T,y),
              0]
    A[1,:] = [v, -1, 0, 0]
    A[2,:] = [0, 0, j0*v, 0]
    A[3,:] = [0, 0, 0, j0*v]
    return A

  def eigA(self, p, h, y, yF, j0):
    ''' Return array of eigenvalues of A, which consists of the pair
      l1 = 0.5 * j0^2 * (dv/dp)_{h,y} -
        0.5 * sqrt((j0^2 dv/dp)^2 + 4(1 + j0^2 dv/dp + v j0^2 (dv/dh)) ),
      l2 = 0.5 * j0^2 * (dv/dp)_{h,y} +
        0.5 * sqrt((j0^2 dv/dp)^2 + 4(1 + j0^2 dv/dp + v j0^2 (dv/dh)) ),
    and
      u, u,
    which transport the chemical state (dy/dx). The conjugate pair eigenvalues
    replace the usual isentropic sound speed eigenvalues.
    ''' 
    # Evaluate mixture state
    T = self.T_ph(p, h, y)
    v = self.v_mix(p, T, y)  
    # Compute eigenvalues
    u = j0 * v
    _q1 = j0**2 * self.dv_dp(p, T, y)
    _q2 = np.sqrt(_q1**2 + 4 * (1 + _q1 + v * j0**2 * self.dv_dh(p, T, y)))
    l1 = 0.5*(_q1 - _q2)
    l2 = 0.5*(_q1 + _q2)
    return np.array([l1, l2, u, u])
  
  def F_fric(self, p, T, y, yF, rho, u, yWt=None, yC=None) -> float:
    ''' Friction (force per unit volume)
    
    Backwards compatible for constant yWt (set yWt to None) or vector
    valued yWt of same size of y.
    Backwards compatible for constant yC (set yC to None) or vector
    valued yC of same size of y.
    '''

    # Poll friction model
    # mu = self.mu
    mu = self.F_fric_viscosity_model(T, y, yF, yWt=yWt, yC=yC)

    # Compute fractional indicator using yF / yM (liquid phase, not liquid melt)
    yM = 1.0 - self.yA - y
    # Continuous alternative to float(self.vf_g(p, T, y) > self.crit_volfrac)

    frag_factor = np.clip(1.0 - yF/yM, 0.0, 1.0)
    return -8.0*mu/(self.conduit_radius*self.conduit_radius) * u * frag_factor

  def F_fric_viscosity_model(self, T, y, yF, yWt=None, yC=None):
    ''' Calculates the viscosity as a function of dissolved
    water and crystal content (assumes crystal phase is incompressible)/
    Does not take into account fragmented vs. not fragmented (avoiding
    double-dipping the effect of fragmentation).

    Backwards compatible for constant yWt (set yWt to None) or vector
    valued yWt of same size of y.
    Backwards compatible for constant yC (set yC to None) or vector
    valued yC of same size of y.
    '''

    # Constant crystal content
    if yWt is None:
      yWt = self.yWt
    if yC is None:
      yC = self.yC

    # Calculate pure melt viscosity (Hess & Dingwell 1996)
    yWd = yWt - y
    yL = self.yL
    yM = 1.0 - y - self.yA
    mfWd = yWd / yL # mass concentration of dissolved water
    mfWd = np.where(mfWd <= 0.0, 1e-8, mfWd)
    log_mfWd = np.log(mfWd*100)
    log10_vis = -3.545 + 0.833 * log_mfWd
    log10_vis += (9601 - 2368 * log_mfWd) / (T - 195.7 - 32.25 * log_mfWd)
    # Prevent overflowing float
    log10_vis = np.where(log10_vis > 300, 300, log10_vis)
    meltVisc = 10**log10_vis
    # Calculate relative viscosity due to crystals (Costa 2005).
    alpha = 0.999916
    phi_cr = 0.673
    gamma = 3.98937
    delta = 16.9386
    B = 2.5
    # Compute volume fraction of crystal at equal phasic densities
    # Using crystal volume per (melt + crystal + dissolved water) volume
    phi_ratio = np.clip((yC / yM) / phi_cr, 0.0, None)
    erf_term = erf(
      np.sqrt(np.pi) / (2 * alpha) * phi_ratio * (1 + phi_ratio**gamma))
    crysVisc = (1 + phi_ratio**delta) * ((1 - alpha * erf_term)**(-B * phi_cr))
    
    viscosity = meltVisc * crysVisc
    return viscosity
  
  def source_y(self, p, y):
    ''' Source term for mass fraction exsolved (y aka yW or yWv) '''
    # Set target mass fraction 
    target_yWd = np.clip(
      self.x_sat(p) * (1.0 - self.yC - self.yWt - self.yA),
      0,
      self.yWt - self.yWvInletMin)
    return 1.0 / self.tau_d * ((self.yWt - y) - target_yWd)

  def choking_ratio(self, p, h, y, j0):
    ''' Compute steady-state analog to u^2 / c^2. Flow is choked
    when the output is equal to 1.'''
    T = self.T_ph(p, h, y)
    return -j0**2 * (self.dv_dp(p, T, y) 
          + self.v_mix(p, T, y) * self.dv_dh(p, T, y))

  def solve_pcoord_system(self, p_chamber, j0):
    ''' Solves initial value problem as (h, y, yf)(p), and then solves the
    pressure gradient equation for dp/dx = F(p, h, y, yf) for the mapping p(x). '''
    yA = self.yA
    # Compute auxiliary inlet conditions
    yWvInlet = np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None)
    h_chamber = self.mixture.h_mix(
      p_chamber, self.T_chamber, self.yA, yWvInlet, 1.0 - self.yA - yWvInlet)
    # Set chamber (inlet) condition (p, h, y) with y = y_eq at pressure
    q0 = np.array([h_chamber, yWvInlet, self.yFInlet])

    ''' Define momentum source (using captured parameters). '''
    # Define gravity momentum source
    F_grav = lambda rho: rho * (-9.8)
    # Define ODE momentum source term sum in terms of density rho
    F_rho = lambda p, T, y, yF, rho: F_grav(rho) \
      + self.F_fric(p, T, y, yF, rho, j0/rho)

    def RHS(p, q, vectorized=False):
      ''' Right-hand side of the primitive variable vector equation 
      with state q == [h, y, yF]. The structure of the mass matrix is
            [B   b]   [_ _ _ 0]                      [_ _ _ 0]
        A = [   uI] = [_ _ 0 0] and inverse sparsity [_ _ _ 0] .
                      [0 0 _ 0]                      [0 0 _ 0]
                      [0 0 0 _]                      [0 0 0 _]
      Note that
        dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
        dv_dh = y * R / p * dT_dh(p, h, y)
        dv_dy = (v_wv(p, T) - v_m(p)) + y * R / p * dT_dy(p, T, y)'''
      p = np.asarray(p)
      # Unpack
      h, y, yF = q
      # Compute dependents
      T     = self.T_ph(p, h, y) 
      v     = self.v_mix(p, T, y) 
      u     = j0 * v
      # Compute first column of B^{-1} with shape (2,1) and value [-1, -v] / det
      a1 = np.vstack((np.ones_like(v), v)) / (1.0 + j0 * j0 * self.dv_dp(p, T, y) \
        + v * j0 * j0 * self.dv_dh(p, T, y))
      # Compute z == -B^{-1} * b / u with shape (2,1)
      z = -j0 * j0 * self.dv_dy(p, T, y) / u * a1
      yL = 1.0 - self.yWt - self.yC - self.yA
      yM = 1.0 - y - self.yA

      vec_length = p.shape[-1] if len(p.shape) > 0 else 1
      # Compute source contribution of exsolution term with mode [[z], 1/u, 0]
      o1 = self.source_y(p,y) * np.vstack((z,  1.0/u, np.zeros_like(u))) \
      # Compute momentum contribution with mode [[a1], 0, 0]
      o2 = F_rho(p, T, y, yF, 1.0/v) * np.vstack((a1, np.zeros((2, vec_length))))
      # Compute fragmentation source contribution with mode [0, 0, 0, 1]
      o3 = np.vstack([np.zeros((3, vec_length)),
           self.frag_source(p, T, y, yF, u * self.tau_f, j0)])
      out = o1 + o2 + o3

      # Transform independent coordinates from x to p
      out = out[1:,...] / out[0:1,...]
      if not vectorized:
        # Return flattened version
        return out.squeeze(axis=-1)
      return out
    
    def RHS_xp(p, q):
      ''' Second ODE set for translating pressure to position. '''
      # Unpack
      p = np.asarray(p)
      h, y, yF = q
      # Compute dependents
      T     = self.T_ph(p, h, y) 
      v     = self.v_mix(p, T, y)
      u     = j0 * v
      # Compute contributions to dp/dx equation from A^{-1} positions (0,0) and (0,2)
      #   A^{-1} has zero at entry (0,3) and source vector has zero at entry (1,).
      neg_det = (1.0 + j0 * j0 * self.dv_dp(p, T, y)
        + v * j0 * j0 * self.dv_dh(p, T, y))
      drag_dpdx = F_rho(p, T, y, yF, 1.0/v) / neg_det
      source_dpdx = self.source_y(p,y) * (
        -j0 * j0 * self.dv_dy(p, T, y) / u / neg_det)
      dxdp = 1.0 / (drag_dpdx + source_dpdx)
      return dxdp
    
    def RHS_reduced(p, h):
      ''' Reduced-size system for j0 == 0 case. (1x1 instead of 3x1).
      Length scale of exsolution and fragmentation -> 0. The reduction is
      simply the energy equation written as
         dh = v dp. '''
      F = np.zeros((1,1))
      # Equilibrium water vapour
      y_eq = self.y_wv_eq(p)
      # Compute mixture temperature
      T = self.T_ph(p, h, y_eq)
      v = self.v_mix(p, T, y_eq)
      # Compute fragmented mass fraction
      yM = 1.0 - y - yA
      yF = yM if self.vf_g(p, T, y) >= self.crit_volfrac else 0
      # Compute source vector with idempotent A^{-1} = A premultiplied
      F[0] = v
      return F.flatten()

    def RHS_xp_reduced(p, h):
      ''' Second ODE set for translating pressure to position. '''
      # Unpack
      p = np.asarray(p)
      # Equilibrium water vapour
      y_eq = self.y_wv_eq(p)
      # Compute mixture temperature
      T = self.T_ph(p, h, y_eq)
      v = self.v_mix(p, T, y_eq)
      # Compute fragmented mass fraction
      yM = 1.0 - y - yA
      yF = yM if self.vf_g(p, T, y) >= self.crit_volfrac else 0
      return 1.0 / F_rho(p, T, y, yF, 1.0/v)

    class EventChokedPCoord():
      def __init__(self, choking_ratio:callable, y_wv_eq=None):
        self.terminal = True
        self.sonic_tol = 0.0
        # Capture function p -> y_wv if provided
        self.y_wv_eq = y_wv_eq
        self.choking_ratio = choking_ratio
      def __call__(self, p, q):
        # Compute equivalent condition to conjugate pair eigenvalue == 0
        # Note that this does not check the condition u == 0 (or j0 == 0).
        if len(q) > 2:
          h, y, yF = q
        else:
          h = q
          y = self.y_wv_eq(p)
        # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
        # dv_dh = y * R / p * dT_dh(p, h, y)
        # Return M^2 == 1 - sonic_tol using captured j0
        return 1.0 - self.sonic_tol - self.choking_ratio(p, h, y, j0)

    p_min = 1e-5

    # Call ODE solver
    if j0 > 0:
      soln_hyy = scipy.integrate.solve_ivp(RHS,
        (p_chamber, p_min),
        q0,
        # t_eval=self.x_mesh,
        method="Radau", dense_output=True,
        events=[EventChokedPCoord(self.choking_ratio)])
      soln_dense_hyy = soln_hyy.sol
      p = soln_hyy.t
      # Compute x
      soln_x = scipy.integrate.solve_ivp(lambda t, y: [RHS_xp(t, soln_dense_hyy(t))],
        (p_chamber, p[-1]),
        [self.x_mesh[0]],
        method="Radau", dense_output=True)
    else:
      # Exact zero flux: use reduced (equilibrium chemistry, fragmentation) system
      soln_hyy = scipy.integrate.solve_ivp(RHS_reduced,
        (p_chamber, p_min),
        q0[0:1],
        # t_eval=self.x_mesh,
        method="Radau", dense_output=True,
        events=[EventChokedPCoord(self.choking_ratio, y_wv_eq=self.y_wv_eq)])
      # Augment output solution with y at equilibrium and yF based on fragmentation criterion
      p = soln_hyy.t
      # Wrap dense solution with equilibrium mass fraction values
      def soln_dense_hyy(t):
        # Interpret independent variable as p
        p = t
        yWv = self.y_wv_eq(p)
        yM = 1.0 - yWv - self.yA
        T = self.T_ph(p, soln_hyy.y[0,:], yWv)
        yF = np.where(self.vf_g(p, T, yWv) >= self.crit_volfrac, yM, 0.0)
        return np.vstack([p, soln_hyy.sol(t), yWv, yF])
      # Compute x
      soln_x = scipy.integrate.solve_ivp(lambda t, y: [RHS_xp(t, soln_dense_hyy(t))],
        (p_chamber, p[-1]),
        [self.x_mesh[0]],
        method="Radau", dense_output=True)

    return {
      "range(p)": (soln_hyy.t.min(), soln_hyy.t.max()),
      "x(p)": soln_x.sol,
      "hyy(p)": soln_dense_hyy,
      "soln_x": soln_x,
    }

  
  def smoother(self, x, scale):
    ''' Returns one-sided smoothing u(x) of a step, such that
      1. u(x < -scale) = 0
      2. u(x >= 0) = 1.
      3. u smoothly interpolates from 0 to 1 in between.
    '''
    # Shift, scale, and clip to [-1, 0] to prevent exp overflow
    _x = np.clip(x / scale + 1, 0, 1)
    f0 = np.exp(-1/np.where(_x == 0, 1, _x))
    f1 = np.exp(-1/np.where(_x == 1, 1, 1-_x))
    # Return piecewise evaluation
    return np.where(_x >= 1, 1,
            np.where(_x <= 0, 0, 
              f0 / (f0 + f1)))

  def frag_source_volfrac(self, p, T, y, yF, L_f, j0):
    ''' Fragmentation source term appearing in ODE for y_f.
      Here L_f is the lengthscale equal to u * tau_f. '''
    yM = 1.0 - self.yA - y
    if self.fragsmooth_scale == 0:
      # Compute rate-limiting-type smoothing factor replacing (yM - yF)
      # rate_factor = np.clip(yM - yF, 0, yF/yM+0.02)
      rate_factor = yM - yF

      return rate_factor / L_f \
        * np.array(self.vf_g(p, T, y) >= self.crit_volfrac).astype(float)
    else:
      # Compute smoothing coordinate and smoothing function
      t = self.vf_g(p, T, y) - self.crit_volfrac
      shape = self.smoother(t, self.fragsmooth_scale)
      return (yM - yF) / L_f * shape
  
  def strain_rate(self, p, T, y, yF, j0):

    def dpdxRHS(p, T, y, yF, vectorized=False):
      ''' Supports vectorized input if vectorized=True. '''
      v     = self.v_mix(p, T, y)
      # Compute first column of A^{-1}:(2,1)
      a1 = 1.0 / (1.0 + j0 * j0 * self.dv_dp(p, T, y) \
                  + v * j0 * j0 * self.dv_dh(p, T, y))
      # Compute z == -A^{-1} * b / u
      z = -j0 * self.dv_dy(p, T, y) / v * a1

      # Define equivalent source for mass fraction exsolved (y, or yWv)
      target_yWd = lambda p: np.clip(
        self.x_sat(p) * (1.0 - self.yC - self.yWt - self.yA), 0, self.yWt - self.yWvInletMin)
      Y = lambda p, y: 1.0 / self.tau_d * ((self.yWt - y) - target_yWd(p))
      # Define gravity momentum source
      F_grav = lambda rho: rho * (-9.8)
      # Define ODE momentum source term sum in terms of density rho
      F_rho = lambda p, T, y, yF, rho: F_grav(rho) \
        + self.F_fric(p, T, y, yF, rho, j0/rho)

      out = Y(p,y) * z + F_rho(p, T, y, yF, 1.0 / v) * a1
      if not vectorized:
        # Return flattened version
        return out.squeeze()
      return out

    # Compute strain rate
    rho = 1.0 / self.v_mix(p, T, y)
    dpdx = dpdxRHS(p, T, y, yF)    
    return -self.rho0_magma * self.rho0_magma / (rho * rho * rho) \
      * j0 / self.K_magma * dpdx

  def frag_source_strain_rate(self, p, T, y, yF, L_f, j0):
    ''' Fragmentation source term appearing in ODE for y_f, using
    a strain-rate based criterion (Papale 99).
      Here L_f is the lengthscale equal to u * tau_f. '''
    yM = 1.0 - self.yA - y

    gamma_dot = self.strain_rate(p, T, y, yF, j0)
    crit_strain_rate = self.crit_strain_rate_k * self.K_magma \
      / self.F_fric_viscosity_model(T, y, yF)

    if self.fragsmooth_scale == 0:
      # Compute rate-limiting-type smoothing factor replacing (yM - yF)
      # rate_factor = np.clip(yM - yF, 0, yF/yM+0.02)
      rate_factor = yM - yF
      
      

      return rate_factor / L_f \
        * np.array(gamma_dot >= crit_strain_rate).astype(float)
    else:
      # Compute smoothing coordinate and smoothing function
      t = gamma_dot - crit_strain_rate
      shape = self.smoother(t, self.fragsmooth_scale * crit_strain_rate)
      return (yM - yF) / L_f * shape

  def solve_ssIVP(self, p_chamber, j0, dense_output=False) -> tuple:
    ''' Solves initial value problem for (p,h,y)(x), given fully specified
    chamber (boundary) state.
    Returns:
      t: array
      state: array (p, h, y) (array sized 3 x ...)
      tup: informational tuple with solve_ivp return value `soln`,
        and system eigvals at vent) '''

    # Pull parameter values
    yA, yWt, yC, crit_volfrac, mu, tau_d, tau_f, conduit_radius, T_chamber = \
      self.yA, self.yWt, self.yC, self.crit_volfrac, self.mu, self.tau_d, \
      self.tau_f, self.conduit_radius, self.T_chamber
    yFInlet = self.yFInlet
    # Compute auxiliary inlet conditions
    yWvInlet = np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None)
    h_chamber = self.mixture.h_mix(
      p_chamber, T_chamber, yA, yWvInlet, 1.0-yA-yWvInlet)

    ''' Define momentum source (using captured parameters). '''
    # Define gravity momentum source
    F_grav = lambda rho: rho * (-9.8)
    # Define ODE momentum source term sum in terms of density rho
    F_rho = lambda p, T, y, yF, rho: F_grav(rho) \
      + self.F_fric(p, T, y, yF, rho, j0/rho)

    ''' Define water vapour mass fraction source. '''
    # Define source in mass per total volume
    S_source = lambda p, y, rho: rho / tau_d * (1.0 - yC - yWt - yA) * float(y > 0) * (
      (yWt - y)/(1.0 - yC - yWt) - self.x_sat(p))
    # Define equivalent source for mass fraction exsolved (y, or yWv)
    target_yWd = lambda p: np.clip(
      self.x_sat(p) * (1.0 - yC - yWt - yA), 0, yWt - self.yWvInletMin)
    Y = lambda p, y: 1.0 / tau_d * ((yWt - y) - target_yWd(p))
    # One-way gating
    # Y = lambda p, y: float(y > 0) * Y_unlimited(p, y) \
    #   + float(y <= 0) * np.clip(Y_unlimited(p,y), 0, None)
    # Ramp gating
    # Y = lambda p, y: np.clip(y, None, self.yWvInletMin) / self.yWvInletMin * Y_unlimited(p, y)
    
    ''' Set source term vector. '''
    def F(q):
      # Unpack q of size (4,1)
      p, h, y, yF = q
      F = np.zeros((4,1))
      # Compute mixture temperature, density
      T = self.T_ph(p, h, y)
      rho = 1.0/self.v_mix(p, T, y)
      # Compute (constant) liquid melt fraction
      yL = 1.0 - self.yWt - self.yC - self.yA
      yM = 1.0 - self.yA - y
      # Compute source vector
      F[0] = F_rho(p, T, y, yF, rho) 
      F[2] = Y(p, y)
      F[3] = self.frag_source(p, T, y, yF, self.tau_f, j0) # hard-coded u = 1
      return F
    if self._DEBUG:  
      # Cache source term (cannot be pickled with default pickle module)
      self.F = F

    ''' Set ODE RHS A^{-1} F '''
    def AinvRHS_numinv(x, q):
      ''' Basic A^{-1} F evaluation.
      Used in ODE solver for dq/dx == RHS(x, q).
      Use AinvRHS instead for speed; this function
      shows more clearly the equation being solved,
      but relies on numerical inversion of a 4x4
      matrix.
      '''
      # Solve for RHS ode
      return np.linalg.solve(self.A(*q, j0), F(q)).flatten()

    def AinvRHS(x, q, vectorized=False):
      ''' Precomputed A^{-1} f for speed.
      Uses block triangular inverse of
        [A b]^{-1}  = [A^{-1}  z ]
        [  u]         [       1/u]
      applied to sparse RHS vector F.
      Use this instead of RHS for speed. Supports vectorized input if
        vectorized=True
      '''
      # Unpack
      p, h, y, yF = q
      # Compute dependents
      T     = self.T_ph(p, h, y)
      # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
      # dv_dh = y * R / p * dT_dh(p, h, y)
      # dv_dy = (v_wv(p, T) - v_m(p)) + y * R / p * dT_dy(p, T, y)
      v     = self.v_mix(p, T, y) 
      u     = j0 * v
      # Compute first column of A^{-1}:(2,1)
      a1 = np.vstack((np.ones_like(v), v)) / (1.0+j0*j0 * self.dv_dp(p, T, y) \
        + v * j0*j0 * self.dv_dh(p, T, y))
      # Compute z == -A^{-1} * b / u
      z = -j0*j0 * self.dv_dy(p, T, y) / u * a1
      # yL = 1.0 - yWt - yC - yA
      yM = 1.0 - y - yA

      vec_length = p.shape[-1] if len(p.shape) > 0 else 1
      # return Y(p, y) * np.array([*z, 1/u, 0]) \
      #   + F_rho(p, T, y, yF, 1.0/v) * np.array([*a1, 0, 0]) \
      #   + np.array([0, 0, 0, (yM - yF) / (u * self.tau_f)
      #     * float(self.vf_g(p, T, y) >= self.crit_volfrac)])
      out = Y(p,y) * np.vstack((z,  1.0/u, np.zeros_like(u))) \
        + F_rho(p, T, y, yF, 1.0/v) \
          * np.vstack((a1, np.zeros((2, vec_length)))) \
        + np.vstack([np.zeros((3, vec_length)),
          self.frag_source(p, T, y, yF, u * tau_f, j0)])
      if not vectorized:
        # Return flattened version
        return out.squeeze(axis=-1)
      return out
    
    def RHS_reduced(x, q):
      ''' Reduced-size system for j0 == 0 case. (2x1 instead of 4x1).
      Length scale of exsolution and fragmentation -> 0. '''
      p, h = q
      F = np.zeros((2,1))
      # Equilibrium water vapour
      y = self.y_wv_eq(p)
      # Compute mixture temperature
      T = self.T_ph(p, h, y)
      v = self.v_mix(p, T, y)
      # Compute fragmented mass fraction
      yM = 1.0 - y - yA
      yF = yM if self.vf_g(p, T, y) >= self.crit_volfrac else 0
      # Compute source vector with idempotent A^{-1} = A premultiplied
      F[0] = 1
      F[1] = v
      F *= F_rho(p, T, y, yF, 1.0/self.v_mix(p, T, y)) 
      return F.flatten()
    
    ''' Define postprocessing eigenvalue checker '''  
    # Set captured lambdas
    T_ph, dv_dp, v_mix, dv_dh = self.T_ph, self.dv_dp, self.v_mix, self.dv_dh
    class EventChoked():
      def __init__(self, y_wv_eq=None):
        self.terminal = True
        self.sonic_tol = 1e-7
        # Capture function p -> y_wv if provided
        self.y_wv_eq = y_wv_eq
      def __call__(self, t, q):
        # Compute equivalent condition to conjugate pair eigenvalue == 0
        # Note that this does not check the condition u == 0 (or j0 == 0).
        if len(q) > 2:
          p, h, y, yF = q
        else:
          p, h = q
          y = self.y_wv_eq(p)
        T = T_ph(p, h, y)
        # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
        # dv_dh = y * R / p * dT_dh(p, h, y)
        return j0**2 * (dv_dp(p, T, y) 
          + v_mix(p, T, y) * dv_dh(p, T, y)) \
          + 1.0 - self.sonic_tol
        # Default numerical eigenvalue computation
        return np.abs(np.linalg.eigvals(A(*q, j0))).min() - self.sonic_tol
    
    class ZeroPressure():
      def __init__(self):
        self.terminal = True
        self.direction = -1.0
      def __call__(self, t, q):
        return q[0] # p

    class ZeroEnthalpy():
      def __init__(self):
        self.terminal = True
        self.direction = -1.0
      def __call__(self, t, q):
        return q[1] # h
    
    class PositivePressureGradient():
      def __init__(self, RHS):
        self.terminal = True
        self.RHS = RHS
      def __call__(self, t, q):
        # Right hand side of dp/dx; is zero when dp/dx>0
        return float(self.RHS(t, q)[0] <= 0)

    # Set chamber (inlet) condition (p, h, y) with y = y_eq at pressure
    q0 = np.array([p_chamber, h_chamber, yWvInlet, yFInlet])

    if self._DEBUG:
      # Cache ODE details
      self.ivp_inputs = (AinvRHS, (self.x_mesh[0],self.x_mesh[-1]), q0, self.x_mesh, "Radau",
        [EventChoked(), ZeroPressure(), ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
    # Call ODE solver
    if j0 > 0:
      soln = scipy.integrate.solve_ivp(AinvRHS,
        (self.x_mesh[0],self.x_mesh[-1]),
        q0,
        # t_eval=self.x_mesh,
        method="Radau", dense_output=dense_output, max_step=5.0,
        events=[EventChoked(), ZeroPressure(),
          ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
      # Output solution
      soln_state = soln.y
    else: # Exsolution length scale u * tau_d -> 0
      # Exact zero flux: use reduced (equilibrium chemistry) system
      soln = scipy.integrate.solve_ivp(RHS_reduced,
        (self.x_mesh[0],self.x_mesh[-1]),
        q0[0:2],
        t_eval=self.x_mesh,
        method="Radau", dense_output=dense_output, max_step=5.0,
        events=[EventChoked(y_wv_eq=self.y_wv_eq), ZeroPressure(),
          ZeroEnthalpy(), PositivePressureGradient(RHS_reduced)])
      # Augment output solution with y at equilibrium and yF based on fragmentation criterion
      p = soln.y[0,:]
      yWv = self.y_wv_eq(p)
      yM = 1.0 - yWv - self.yA
      T = self.T_ph(p, soln.y[1,:], yWv)
      yF = yM.copy()
      yF = np.where(self.vf_g(p, T, yWv) >= self.crit_volfrac, yM, 0.0)
      soln_state = np.vstack((soln.y, yWv, yF))

    # Compute eigenvalues at the final t
    eigvals_t_final = self.eigA(*soln_state[:,-1], j0)

    return soln.t, soln_state, (soln, eigvals_t_final)


  def solve_ssIVP_yC_profile(self, p_chamber, j0, yC_fn:callable,
                             yWt_fn:callable, dense_output=False) -> tuple:
    ''' Solves initial value problem for (p,h,y)(x), given fully specified
    chamber (boundary) state and a profile for yC and yWt. The args
    yC and yWt are functions that return crystal and total water mass fractions
    as a functon of space.
    Returns a tuple with elements:
      t: array
      state: array (p, h, y) (array sized 3 x ...)
      tup: informational tuple with solve_ivp return value `soln`,
        and system eigvals at vent) '''

    # Pull parameter values
    yA, crit_volfrac, mu, tau_d, tau_f, conduit_radius, T_chamber = \
      self.yA, self.crit_volfrac, self.mu, self.tau_d, \
      self.tau_f, self.conduit_radius, self.T_chamber
    yFInlet = self.yFInlet
    # Compute auxiliary inlet conditions
    yWtInlet = yWt_fn(self.x_mesh[0])
    yWvInlet = np.clip(self.y_wv_eq(p_chamber, yWtInlet), self.yWvInletMin, None)
    h_chamber = self.mixture.h_mix(
      p_chamber, T_chamber, yA, yWvInlet, 1.0 - yA - yWvInlet)

    ''' Define momentum source (using captured parameters). '''
    # Define gravity momentum source
    F_grav = lambda rho: rho * (-9.8)
    # Define ODE momentum source term sum in terms of density rho
    F_rho = lambda p, T, y, yF, rho, yWt, yC: F_grav(rho) \
      + self.F_fric(p, T, y, yF, rho, j0/rho, yWt, yC)

    ''' Define water vapour mass fraction source. '''
    target_yWd = lambda p, yWt, yC: np.clip(
      self.x_sat(p) * (1.0 - yC - yWt - yA),
      0, yWt - self.yWvInletMin)
    Y = lambda p, y, yWt, yC: 1.0 / tau_d * ((yWt - y) - target_yWd(p, yWt, yC))
    
    ''' Check debug state for caching RHS vector. '''
    if self._DEBUG:  
      def F(q):
        # Unpack q of size (4,1)
        p, h, y, yF = q
        F = np.zeros((4,1))
        # Compute mixture temperature, density
        T = self.T_ph(p, h, y)
        rho = 1.0/self.v_mix(p, T, y)
        # Compute (constant) liquid melt fraction
        yL = 1.0 - self.yWt - self.yC - self.yA
        yM = 1.0 - self.yA - y
        # Compute source vector
        F[0] = F_rho(p, T, y, yF, rho) 
        F[2] = Y(p, y)
        F[3] = self.frag_source(p, T, y, yF, self.tau_f, j0) # hard-coded u = 1
        return F
      # Cache source term (cannot be pickled with default pickle module)
      self.F = F

    def AinvRHS(x, q, vectorized=False):
      ''' Precomputed A^{-1} f for speed.
      Uses block triangular inverse of
        [A b]^{-1}  = [A^{-1}  z ]
        [  u]         [       1/u]
      applied to sparse RHS vector F.
      Use this instead of RHS for speed. Supports vectorized input if
        vectorized=True
      '''
      # Unpack
      p, h, y, yF = q
      # Evaluate spatially dependent yWt, yC
      yWt, yC = yWt_fn(x), yC_fn(x)
      # Compute dependents
      T     = self.T_ph(p, h, y)
      # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
      # dv_dh = y * R / p * dT_dh(p, h, y)
      # dv_dy = (v_wv(p, T) - v_m(p)) + y * R / p * dT_dy(p, T, y)
      v     = self.v_mix(p, T, y) 
      u     = j0 * v
      # Compute first column of A^{-1}:(2,1)
      a1 = np.vstack((np.ones_like(v), v)) / (1.0 + j0*j0*self.dv_dp(p, T, y) \
        + v * j0*j0*self.dv_dh(p, T, y))
      # Compute z == -A^{-1} * b / u
      z = -j0*j0 * self.dv_dy(p, T, y) / u * a1
      # yL = 1.0 - yWt - yC - yA
      yM = 1.0 - y - yA

      vec_length = p.shape[-1] if len(p.shape) > 0 else 1
      # return Y(p, y) * np.array([*z, 1/u, 0]) \
      #   + F_rho(p, T, y, yF, 1.0/v) * np.array([*a1, 0, 0]) \
      #   + np.array([0, 0, 0, (yM - yF) / (u * self.tau_f)
      #     * float(self.vf_g(p, T, y) >= self.crit_volfrac)])
      out = Y(p, y, yWt, yC) * np.vstack((z,  1.0/u, np.zeros_like(u))) \
        + F_rho(p, T, y, yF, 1.0/v, yWt, yC) \
          * np.vstack((a1, np.zeros((2, vec_length)))) \
        + np.vstack([np.zeros((3, vec_length)),
          self.frag_source(p, T, y, yF, u * tau_f, j0)])
      if not vectorized:
        # Return flattened version
        return out.squeeze(axis=-1)
      return out
    
    def RHS_reduced(x, q):
      ''' Reduced-size system for j0 == 0 case. (2x1 instead of 4x1).
      Length scale of exsolution and fragmentation -> 0. '''
      raise NotImplementedError("Not updated reduced model.")
      p, h = q
      F = np.zeros((2,1))
      # Equilibrium water vapour
      y = self.y_wv_eq(p)
      # Compute mixture temperature
      T = self.T_ph(p, h, y)
      v = self.v_mix(p, T, y)
      # Compute fragmented mass fraction
      yM = 1.0 - y - yA
      yF = yM if self.vf_g(p, T, y) >= self.crit_volfrac else 0
      # Compute source vector with idempotent A^{-1} = A premultiplied
      F[0] = 1
      F[1] = v
      F *= F_rho(p, T, y, yF, 1.0/self.v_mix(p, T, y)) 
      return F.flatten()
    
    ''' Define postprocessing eigenvalue checker '''  
    # Set captured lambdas
    T_ph, dv_dp, v_mix, dv_dh = self.T_ph, self.dv_dp, self.v_mix, self.dv_dh
    class EventChoked():
      def __init__(self, y_wv_eq=None):
        self.terminal = True
        self.sonic_tol = 1e-7
        # Capture function p -> y_wv if provided
        self.y_wv_eq = y_wv_eq
      def __call__(self, t, q):
        # Compute equivalent condition to conjugate pair eigenvalue == 0
        # Note that this does not check the condition u == 0 (or j0 == 0).
        if len(q) > 2:
          p, h, y, yF = q
        else:
          p, h = q
          y = self.y_wv_eq(p)
        T = T_ph(p, h, y)
        # dv_dp = y * (R / p * dT_dp(p, h, y) - R * T / p**2) + (1 - y) * dvm_dp(p)
        # dv_dh = y * R / p * dT_dh(p, h, y)
        return j0**2 * (dv_dp(p, T, y) 
          + v_mix(p, T, y) * dv_dh(p, T, y)) \
          + 1.0 - self.sonic_tol
        # Default numerical eigenvalue computation
        return np.abs(np.linalg.eigvals(A(*q, j0))).min() - self.sonic_tol
    
    class ZeroPressure():
      def __init__(self):
        self.terminal = True
        self.direction = -1.0
      def __call__(self, t, q):
        return q[0] # p

    class ZeroEnthalpy():
      def __init__(self):
        self.terminal = True
        self.direction = -1.0
      def __call__(self, t, q):
        return q[1] # h
    
    class PositivePressureGradient():
      def __init__(self, RHS):
        self.terminal = True
        self.RHS = RHS
      def __call__(self, t, q):
        # Right hand side of dp/dx; is zero when dp/dx>0
        return float(self.RHS(t, q)[0] <= 0)

    # Set chamber (inlet) condition (p, h, y) with y = y_eq at pressure
    q0 = np.array([p_chamber, h_chamber, yWvInlet, yFInlet])

    if self._DEBUG:
      # Cache ODE details
      self.ivp_inputs = (AinvRHS, (self.x_mesh[0],self.x_mesh[-1]), q0, self.x_mesh, "Radau",
        [EventChoked(), ZeroPressure(), ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
    # Call ODE solver
    if j0 > 0:
      soln = scipy.integrate.solve_ivp(AinvRHS,
        (self.x_mesh[0],self.x_mesh[-1]),
        q0,
        # t_eval=self.x_mesh,
        method="Radau", dense_output=dense_output, max_step=10.0, # changed from 5.0
        events=[EventChoked(), ZeroPressure(),
          ZeroEnthalpy(), PositivePressureGradient(AinvRHS)])
      # Output solution
      soln_state = soln.y
    else: # Exsolution length scale u * tau_d -> 0
      # Exact zero flux: use reduced (equilibrium chemistry) system
      soln = scipy.integrate.solve_ivp(RHS_reduced,
        (self.x_mesh[0],self.x_mesh[-1]),
        q0[0:2],
        t_eval=self.x_mesh,
        method="Radau", dense_output=dense_output, max_step=5.0,
        events=[EventChoked(y_wv_eq=self.y_wv_eq), ZeroPressure(),
          ZeroEnthalpy(), PositivePressureGradient(RHS_reduced)])
      # Augment output solution with y at equilibrium and yF based on fragmentation criterion
      p = soln.y[0,:]
      yC = yC_fn(soln.x)
      yWt = yWt_fn(soln.x)
      yWv = self.y_wv_eq(p)
      yM = 1.0 - yWv - self.yA
      T = self.T_ph(p, soln.y[1,:], yWv)
      yF = yM.copy()
      yF = np.where(self.vf_g(p, T, yWv) >= self.crit_volfrac, yM, 0.0)
      soln_state = np.vstack((soln.y, yWv, yF))

    # Compute eigenvalues at the final t
    eigvals_t_final = self.eigA(*soln_state[:,-1], j0)

    return soln.t, soln_state, (soln, eigvals_t_final)


  def _set_cache(self, p_vent:float, inlet_input_val:float,
    input_type:str="u"):
    _, _, calc_details = self.solve_steady_state_problem(
      p_vent, inlet_input_val, input_type=input_type, verbose=True)
    self.j0 = calc_details["j0"]
    self.p_chamber = calc_details["p"]
  
  def __call__(self, x:np.array, io_format:str="quail") -> np.array:
    '''Returns U sampled on x in quail format (default).
    Requires x to be points in interval [self.x_mesh.min(), self.x_mesh.max()].
    Inputs:
      x: array of points. If io_format=="quail", x is expected to have
        three-dimensional shape (ne, n).
      io_format: either "quail" or "phy". The latter is the native ODE solver
        output in (p, h, y, yF) space. Here y 
    ''' 
    # Check that input x is consistent with internal length
    if x.max() > self.x_mesh.max() \
      or x.min() < self.x_mesh.min():
      raise ValueError("Requested values at x not in initial global mesh.")

    # Solve steady state IVP with native mesh and precomputed mass flux
    soln = self.solve_ssIVP(self.p_chamber, self.j0, dense_output=True)
    # Extract interpolator from scipy.integrate.solve_ivp
    dense_soln = soln[2][0].sol
    # Evaluate solution using interpolator
    Q = dense_soln(np.unique(x))
    # Extrapolate out-of-bounds values using nearest value
    last_legit_index = len(np.unique(x)) \
      - np.argmax(np.unique(x)[::-1] <= soln[0].max()) - 1
    Q[:, np.unique(x) > soln[0].max()] = \
      Q[:, last_legit_index:last_legit_index+1]

    # Compute solution in requested format
    if "phy".casefold() == io_format.casefold() \
       or "native".casefold() == io_format.casefold():
      # Return solution state (p, h, y)
       # Evaluate solution using interpolator
      return Q
    elif "quail".casefold() == io_format.casefold():
      p, h, y, yF = Q
      # Mass fraction correction
      y = np.where(y < 0, self.yWvInletMin, y)
      # Crystallinity correction
      yC = np.max((self.yC, self.yCMin))
      # Compute mixture intermediates
      T = self.T_ph(p, h, y)
      v = self.v_mix(p, T, y)
      # Load and return conservative state vector
      U = np.zeros((*np.unique(x).shape,8))
      U[...,0] = self.yA / v
      U[...,1] = y / v
      U[...,2] = (1.0 - y - self.yA) / v
      U[...,3] = self.j0
      U[...,4] = 0.5 * self.j0**2 * v + h/v - p
      U[...,5] = self.yWt / v
      U[...,6] = yC / v
      U[...,7] = yF / v

      ''' Extract only values of U that correspond to query locations x. '''
      # Define associative map from value of x to state vector U
      vals = {x: U[i,:] for i, x in enumerate(np.unique(x))}
      U_out = np.zeros((*x.shape[:2],8))
      # Map sample locations to state values U
      for i in range(U_out.shape[0]):
        for j in range(U_out.shape[1]):
          U_out[i,j,:] = vals[x[i,j,0]]
      return U_out
    else:
      raise ValueError(f"Unknown output format string '{io_format}'.")

  def solve_steady_state_problem(self, p_vent:float, inlet_input_val:float,
    input_type:str="u", verbose=False, _internal_debug_flag=False):
    ''' Solves for the choking pressure and corresponding flow state.
    Input mass flux j0, velocity u, or chamber pressure p_chamber to compute
    the corresponding steady state. Specify input_type="j", "u", or "p" to access
    these modes. Note that j, u, and p are interdependent so that only one
    is required.
    The steady state problem is solved using the shooting method for j0 or p
    against the prescribed vent pressure if the pressure is above the choking
    pressure but below the hydrostatic vent pressure.

    Devnote: The p-case can be made faster by directly solving for the choking
    pressure, and then checking if flow is choked at the computed j0.
    '''
    # Pull parameter values
    yA, yWt, yC, crit_volfrac, mu, tau_d, tau_f, conduit_radius, T_chamber = \
      self.yA, self.yWt, self.yC, self.crit_volfrac, self.mu, self.tau_d, \
      self.tau_f, self.conduit_radius, self.T_chamber
    
    p_global_min = 0.1e5
    if p_vent < p_global_min:
      raise ValueError("Vent pressure below lowest tested case (0.1 bar).")

    # Select mode
    if input_type.lower() in ("u", "u0",):
      # Set p_chamber range for finding max
      p_min, p_max = np.max((p_global_min, p_vent)), 1e9
      z_min, z_max = p_min, p_max
      # Define dependence of inlet volume on p_chamber
      v_pc = lambda p_chamber: self.v_mix(p_chamber, self.T_chamber,
          np.clip(self.y_wv_eq(p_chamber), self.yWvInletMin, None))
      
      u0 = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda p_chamber: self.solve_ssIVP(
        p_chamber, u0/v_pc(p_chamber))
      p_vent_max = p_max
      _input_type = "u"
      mass_flux_cofactor = lambda p: 1.0/v_pc(p)

      ''' Additional estimation to filter out root for fragmented magma at the inlet '''
      # Compute maximum exsolvable in conduit
      yMax = self.yWt - self.x_sat(p_vent) * (1.0 - self.yC - self.yWt - self.yA)
      # Compute maximum water vapour volume
      vwMax = 1.0 / p_vent * self.mixture.waterEx.R * self.T_chamber
      # Compute maximum mixture volume
      vMax = yMax * vwMax + (1 - yMax) * self.mixture.magma.v_pT(p_vent, None)
      # Estimate minimum chamber pressure
      p_est = p_vent + self.conduit_length * 9.8 /  vMax
      # Compute saturation pressure
      p_sat = (self.yWt / self.yL / self.solubility_k) ** (1/self.solubility_n)
      p_min = np.max((p_min, p_sat))
      # print(p_est, p_min, p_sat)
      z_min = p_min

    elif input_type.lower() in ("j", "j0",):
      # Set p_chamber range for finding max
      p_min, p_max = np.max((p_global_min, p_vent)), 1e9
      z_min, z_max = p_min, p_max
      j0 = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda p_chamber: self.solve_ssIVP(
        p_chamber, j0)
      p_vent_max = p_max
      _input_type = "u"
      mass_flux_cofactor = lambda p: 1.0
    elif input_type.lower() in ("p", "p0",):
      # Set j0 range for finding max 
      j0_min, j0_max  = 0.0, 2.7e3*100
      z_min, z_max = j0_min, j0_max
      p_chamber = inlet_input_val
      # Define solve kernel that returns (x, (p, h, y), (soln, eigvals))
      solve_kernel = lambda j0: self.solve_ssIVP(p_chamber, j0)
      p_vent_max = solve_kernel(j0_min)[1][0][-1]
      _input_type = "p"
    else:
      raise Exception('Unknown input_type (use "u", "j", "p").')
    
    # Define mapping z -> p_vent, where z is the conjugate to the input value
    # (user inputs u or j0: z is p; user inputs p: z is j0)
    calc_vent_p = lambda z: solve_kernel(z)[1][0][-1]
    # Define mapping z -> lambda_min(x = 0)
    def eigmin_top(z):
      ''' Returns the smaller conjugate-pair eigval at top,
      or negative value if the matrix is singular at depth.
      Assumes that the correct eigval is indexed by 1 in list
      [u-k, u+k, u, u].''' 
      _t, _z, outs = solve_kernel(z)
      return -1e-1 if len(outs[0].t_events[0]) != 0 or not outs[0].success else \
        np.abs(outs[1][0])
    
    def guarded_p_top_diff(z):
      ''' Returns pressure at top of domain, with guard for internal choking.''' 
      _t, _z, outs = solve_kernel(z)
      return -1e-1 if len(outs[0].t_events[0]) != 0 or not outs[0].success else \
        _z[0,-1]

    ''' 
    For input p_chamber, the bounds on p_vent are given by the hydrostatic and
    choking j0.
    For input u or j, for low enough chamber pressure, p drops below p_vent. For
    high enough chamber pressure, the flow chokes at the vent but vent pressure
    is continuously dependent on the chamber pressure. The hydrostatic p_chamber
    provides a lower bound on p_chamber. As p_chamber increases, p_vent.
    '''
    # Solve for maximum j0 / minimum p_chamber that does not choke
    
    brent_atol = self.brent_atol
    if _input_type == "p":
      z_choke = scipy.optimize.brentq(lambda z: eigmin_top(z),
        z_min, z_max, xtol=brent_atol)
      # Bias to get slightly unchoked within tolerance
      print(f"{calc_vent_p(z_choke-2*brent_atol)} and {calc_vent_p(z_choke+2*brent_atol)}")
      print(f"{eigmin_top(z_choke-2*brent_atol)} and {eigmin_top(z_choke+2*brent_atol)}")

      z_min = z_choke
      ''' Check vent flow state for given p_vent, and solve for solution
      [p(x), h(x), y_i(x)] where y_i are the mass fractions. '''
      p_choke = calc_vent_p(z_choke)
      print(f"Computed choking mass flux: {z_choke}; " + 
            f"choking pressure: {p_choke}.")
      if p_vent < p_choke:
        # Choked case
        print(f"Choked at vent.")
        # Solve with one-sided precision to ensure that the last node is
        # evaluable (i.e., choking position is >= top). This is not a guarantee
        # when solution is requested 
        z = z_choke - 2*brent_atol
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = \
          solve_kernel(z)
      elif p_vent > p_vent_max:
        # Inconsistent pressure (exceeds hydrostatic pressure consistent with chamber pressure)
        print("Vent pressure is too high (reverse flow required to reverse pressure gradient).")
        x, (p_soln, h_soln, y_soln, yF_soln), soln = None, (None, None, None), None
      else:
        print("Subsonic flow at vent. Shooting method for correct value of z.")
        if _internal_debug_flag:
          return lambda z: calc_vent_p(z) - p_vent, z_min, z_max
        # Set search range j0 in [0.0, z_min==z_choke==j0_choked]
        # For mass flux below choking, flow should be subsonic in the interior
        z = scipy.optimize.brentq(lambda z: calc_vent_p(z) - p_vent, 0.0, z_min, xtol=brent_atol)
        print("Solution j0 found. Computing solution.")
        # Compute solution at j0
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = solve_kernel(z)
    elif _input_type == "u":
      # Number of times to double pressure while searching for choking pressure
      N_doubling = 14

      # Express mass flux j0 given u
      if input_type.lower() in ("j", "j0",):        
        j0_u = lambda p: j0
      else:
        u = inlet_input_val
        j0_u = lambda p: u / self.v_mix(p, self.T_chamber,
          np.clip(self.y_wv_eq(p), self.yWvInletMin, None))

      print("Computing lower bound on pressure given domain length.")
      ''' Compute loose lower bound on pressure due to gravity. '''
      # Lower bound pressure
      yMax = self.yWt - self.x_sat(p_vent) * (
        1.0 - self.yC - self.yWt - self.yA)
      # Compute maximum water vapour volume
      vwMax = 1.0 / p_vent * self.mixture.waterEx.R * self.T_chamber
      # Compute maximum mixture volume
      vMax = yMax * vwMax + (1 - yMax) * self.mixture.magma.v_pT(p_vent, None)
      p_minbound = p_vent + self.conduit_length * 9.8 /  vMax

      ''' Find lowest-pressure continuous solution '''
      _use_legacy_bound_method = False
      if _use_legacy_bound_method:
        # Arbitrarily set minimum pressure
        # approx_pseudogas_scale_height = (self.yWt * self.mixture.waterEx.R
        #   * self.T_chamber) / 9.8
        # p_guess = p_vent * np.exp(self.conduit_length /
        #   approx_pseudogas_scale_height)
        # p0 = 10*p_guess
        p0 = p_minbound
        k_last = None
        for k in range(N_doubling):
          # Compute IVP solution
          
          p_chamber = p0*2**k
          j0 = j0_u(p_chamber)
          _out = self.solve_ssIVP(p0*2**k, j0)
          p_top = _out[1][0,-1]
          x_top = _out[0][-1]
          # ''' Top pressure and pressure grad check'''
          # q_top = _out[1][:,-1]
          # dqdx = np.linalg.solve(self.A(*q_top, j0), self.F(q_top)).flatten()
          # dpdx = dqdx[0]
          # # Tolerable distance-to-zero-pressure
          # p_min = 0.001e6 # 1 mbar
          # dx_min = 1.0    # 1 m until zero pressure
          # is_reached_vacuum = p_top < p_min or p_top/np.abs(dpdx) < dx_min
          
          # Search criterion
          if x_top >= self.x_mesh[-1]:
            # Register k
            k_last = k
            # break
        if k_last is None:
          raise Exception("Could not bracket lower pressure limit.")

      ''' Sample function p_vent(p_chamber) to find highest-pressure choke. '''
      # Companion function: p_vent(p_chamber)
      search_p = np.linspace(p_minbound, 300e6, 50)
      def penalized_top_pressure(p_chamber):
        soln = self.solve_ssIVP(p_chamber, j0_u(p_chamber))
        if soln[0][-1] < self.x_mesh[-1]:
          return -1
        return soln[1][0, -1]
      tentatives_p_vent = [penalized_top_pressure(p) for p in search_p] 
      # (debug) plot p_chamber to p_vent mapping
      # plt.semilogy(search_p, tentatives_p_vent)
      # plt.xlabel("Inlet pressure (Pa)")
      # plt.ylabel("Vent pressure (Pa)")
      _i = len(tentatives_p_vent) - np.argmax(
        np.array(tentatives_p_vent[::-1]) < 0)

      def bisect_pmin(a,b):
        ''' Manual bisection for minimum pressure. '''
        fn_x_top = lambda p: self.solve_ssIVP(p, j0_u(p))[0][-1]
        # Reject if continuous solution at low bracketing pressure
        if fn_x_top(a) >= self.x_mesh[-1]:
          raise ValueError(f"Pressure {a} is a continuous solution.")
        # Reject if no continuous solution at high bracketing pressure
        if fn_x_top(b) < self.x_mesh[-1]:
          raise ValueError(f"Pressure {b} is not a continuous solution.")

        m = 0.5*(a+b)

        while b - a > brent_atol:
          # Search criterion
          if fn_x_top(m) >= self.x_mesh[-1]: # Continuous solution found
            b = m
          else:
            a = m
          m = 0.5*(a+b)
        return b # Continuous solution

      # Compute minimum possible chamber pressure
      # pc_min = bisect_pmin(p0*2**k/2, p0*2**k)
      pc_min = bisect_pmin(search_p[_i-1], search_p[_i])

      # Compute corresponding minimum vent pressure
      p_vent_min = self.solve_ssIVP(pc_min, j0_u(pc_min))[1][0,-1]
      self._check = (pc_min, j0_u(pc_min))
      print(f"Minimum vent pressure is {p_vent_min}.")
      if p_vent <= p_vent_min:
        print("Choked flow.")
        z = pc_min
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = solve_kernel(z)
      else:
        # Unchoked
        print("Pressure matching for vent pressure at given velocity.")
        # Define wrapped objective taking into subpressurized flow
        def objective(z):
          soln = solve_kernel(z)
          # Retrieve 
          z_top = soln[0][-1]
          p_top = soln[1][0,-1]
          return p_top - p_vent if z_top >= self.x_mesh[-1] else -p_vent
        z = scipy.optimize.brentq(objective, pc_min, z_max, xtol=brent_atol)
        x, (p_soln, h_soln, y_soln, yF_soln), (soln, _) = solve_kernel(z)

    if verbose:
      # Package extra details on calculation.
      calc_details = {
        "soln": soln,
      }
      if _input_type == "u":
        calc_details["p_min"] = z_min
        calc_details["p_max"] = z_max
        calc_details["p"] = z
        # Mass flux cofactor is rho if input was u rather than rho*u
        calc_details["j0"] = mass_flux_cofactor(z) * inlet_input_val
      elif _input_type == "p":
        calc_details["j0_min"] = z_min
        calc_details["j0_max"] = z_max
        calc_details["p"] = inlet_input_val
        calc_details["j0"] = z

      return x, (p_soln, h_soln, y_soln, yF_soln), calc_details
    else:
      return x, (p_soln, h_soln, y_soln, yF_soln)


class StaticPlug():

  def __init__(self, x_global:np.array, p_chamber:float, 
               traction_fn:callable, yWt_fn:callable, yC_fn:callable, T_fn:callable,
               yF_fn:callable=None,
               override_properties:dict=None, enforce_p_vent=None, num_state_variables=8):
    ''' 
    Hydrostatic ODE solver with plug.
    Inputs:
      x_global: global x coordinates (necessary to coordinate solution between
        several 1D patches)
      p_chamber:        chamber pressure (Pa)
      traction_fn:      function prescribing initial traction
      yWt_fn:           function prescribing yWt(x)
      yC_fn:            function prescribing yC(x)
      T_fn:             function prescribing T(x)
      yF_fn (optional): function prescribing yF(x)
      override_properties (dict): map from property name to value. See first
        section of StaticPlug.__init__ for overridable properties.
      num_state_variables: number of state variables in the system. Default is 8.

    Call this object to sample the solution at a grid x, consistent with the
    provided value of conduit_length.
    '''

    # Update function-valued properties
    self.traction_fn = traction_fn
    self.yWt_fn = yWt_fn
    self.yC_fn = yC_fn
    self.num_state_variables = num_state_variables
    if yF_fn is None:
      # Default yF_fn is uniform 0
      self.yF_fn = lambda x: np.zeros_like(x)
    else:
      self.yF_fn = yF_fn
    self.T_fn = T_fn
    self.p_chamber = p_chamber
    # Validate properties
    if override_properties is None:
      override_properties = {}
    self.override_properties = override_properties.copy()
    ''' Set default and overridable properties'''
    self.yA             = self.override_properties.pop("yA", 1e-7) # yA > 0 for positivity preserving limiter
    self.c_v_magma      = self.override_properties.pop("c_v_magma", 3e3)
    self.rho0_magma     = self.override_properties.pop("rho0_magma", 2.7e3)
    self.K_magma        = self.override_properties.pop("K_magma", 10e9)
    self.p0_magma       = self.override_properties.pop("p0_magma", 5e6)
    self.solubility_k   = self.override_properties.pop("solubility_k", 5e-6)
    self.solubility_n   = self.override_properties.pop("solubility_n", 0.5)
    # Whether to neglect deformation energy in magma EOS
    self.neglect_edfm   = self.override_properties.pop("neglect_edfm", False)
    self.fragmentation_criterion = self.override_properties.pop("fragmentation_criterion", "VolumeFraction")

    # Save option / value for enforcing vent pressure
    self.enforce_p_vent = enforce_p_vent

    # Bind fragmentation source
    if self.fragmentation_criterion.casefold() == "VolumeFraction".casefold():
      pass # Removed. See class SteadyState
      # self.frag_source = self.frag_source_volfrac
    elif self.fragmentation_criterion.casefold() == "StrainRate".casefold():
      pass  # Removed. See class SteadyState
      # self.frag_source = self.frag_source_strain_rate
    else:
      raise ValueError(f"Unknown fragmentation criterion with string " +
                       f"{self.fragmentation_criterion}. Try " +
                       f"VolumeFraction, or StrainRate.")

    # Output mesh
    self.x_mesh = x_global.copy()
    # Internal computation mesh
    self.x_mesh_native = x_global.copy()
    self.conduit_length = x_global.max() - x_global.min()

    # Input validation
    if len(self.override_properties.items()) > 0:
      raise ValueError(
        f"Unused override properties:{list(self.override_properties.keys())}")

    # Set depth of conduit inlet
    self.x0 = x_global.min()

    # Set mixture properties
    mixture = matprops.MixtureMeltCrystalWaterAir()
    mixture.magma = matprops.MagmaLinearizedDensity(c_v=self.c_v_magma,
      rho0=self.rho0_magma, K=self.K_magma,
      p_ref=self.p0_magma, neglect_edfm=self.neglect_edfm)
    mixture.k, mixture.n = self.solubility_k, self.solubility_n
    self.mixture = mixture

    self._has_lambda = True

  ''' Define partially evaluated thermo functions in terms of (p, h, y)'''

  def T_ph(self, p, h, y):
    return self.mixture.T_ph(p, h, self.yA, y, 1.0-y-self.yA)

  def v_mix(self, p, T, y):
    return self.mixture.v_mix(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dp(self, p, T, y):
    return self.mixture.dv_dp(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dh(self, p, T, y):
    return self.mixture.dv_dh(p, T, self.yA, y, 1.0-y-self.yA)

  def dv_dy(self, p, T, y):
    return self.mixture.dv_dy(p, T, self.yA, y, 1.0-y-self.yA)

  def vf_g (self, p, T, y):
    return self.mixture.vf_g(p, T, self.yA, y, 1.0-y-self.yA)

  def x_sat(self, p):
    return self.mixture.x_sat(p)

  def y_wv_eq(self, p):
    return self.mixture.y_wv_eq(p, self.yWt, self.yC)
  
  def __call__(self, x:np.array, io_format:str="quail",
               p_vent_atol=1e-3, p_vent_max_num_iterations:int=16,
               is_solve_direction_downward=False) -> np.array:
    '''Returns U sampled on x in quail format (default).
    Requires x to be points in interval [self.x_mesh.min(), self.x_mesh.max()].
    Inputs:
      x: array of points. If io_format=="quail", x is expected to have
        three-dimensional shape (ne, n).
      io_format: either "quail" or "p". The latter is only pressure.
      p_vent_atol: absolute tolerance in p_vent error for scaling traction if
        p_vent is provided through self.enforce_p_vent
      p_vent_max_num_iterations: max number of iterations to use for matching p_vent
      is_solve_direction_downward (default: False): if True, ignore self.p_chamber,
        and solve initial value problem from vent downward using self.enforce_p_vent
    ''' 

    # Check validity
    if not self._has_lambda:
      raise ValueError("This function has been called already, and the callables"
                       + " traction_fn, yWt_fn, yC_fn, yF_fn, T_fn have been deleted for"
                       + " pickling in quail. To disable this behaviour, see"
                       + " in module steady_state StaticPlug.__call__.")


    # Check that input x is consistent with internal length
    if x.max() > self.x_mesh.max() \
      or x.min() < self.x_mesh.min():
      raise ValueError("Requested values at x not in initial global mesh.")

    # Check for p_vent scale option
    if is_solve_direction_downward:
      if self.enforce_p_vent is None or self.enforce_p_vent == 0:
        raise ValueError("Did not find a nonzero value for p_vent. "
                         +"Reinitialize this object with a positive value for enforce_p_vent. ")
      # Define hydrostatic RHS
      def hydrostatic_RHS(x, p):
        yWt = self.yWt_fn(x)
        yC = self.yC_fn(x)
        T = self.T_fn(x)
        yWv = self.mixture.y_wv_eq(p, yWt, yC)
        return self.traction_fn(x) - 9.8 / self.v_mix(p, T, yWv)
      soln = scipy.integrate.solve_ivp(hydrostatic_RHS,
          (self.x_mesh[-1],self.x_mesh[0]), # This goes (x_top, x_bottom)
          np.array([self.enforce_p_vent]),
          # t_eval=self.x_mesh,
          # method="Radau",
          max_step=0.5, # Likely mesh size in Quail
          dense_output=True,)
    elif self.enforce_p_vent is None or self.enforce_p_vent == 0:
      # Define hydrostatic RHS
      def hydrostatic_RHS(x, p):
        yWt = self.yWt_fn(x)
        yC = self.yC_fn(x)
        T = self.T_fn(x)
        yWv = self.mixture.y_wv_eq(p, yWt, yC)
        return self.traction_fn(x) - 9.8 / self.v_mix(p, T, yWv)
      soln = scipy.integrate.solve_ivp(hydrostatic_RHS,
          (self.x_mesh[0],self.x_mesh[-1]),
          np.array([self.p_chamber]),
          # t_eval=self.x_mesh,
          # method="Radau",
          max_step=0.5, # Likely mesh size in Quail
          dense_output=True,)
    else:
      # Define hydrostatic RHS
      def tractionfree_hydrostatic_RHS(x, p):
        yWt = self.yWt_fn(x)
        yC = self.yC_fn(x)
        T = self.T_fn(x)
        yWv = self.mixture.y_wv_eq(p, yWt, yC)
        return - 9.8 / self.v_mix(p, T, yWv)
      soln_tractionfree = scipy.integrate.solve_ivp(tractionfree_hydrostatic_RHS,
          (self.x_mesh[0],self.x_mesh[-1]),
          np.array([self.p_chamber]),
          # t_eval=self.x_mesh,
          # method="Radau",
          max_step=0.5, # Likely mesh size in Quail
          dense_output=True,)
      # Extract initial p_vent
      p_vent_iterate = soln_tractionfree.y[0,-1]
      # Integrate traction_fn using trapezoidal method
      delta_p = -scipy.integrate.trapezoid(self.traction_fn(np.unique(x)), np.unique(x))
      if delta_p < 0:
        raise ValueError(f"Positive traction, indicating upward drag on static fluid in conduit.")
      scale_value = (p_vent_iterate - self.enforce_p_vent) / delta_p
      if scale_value < 0:
        raise ValueError(f"Pressure without traction ({p_vent_iterate}) is below specified p_vent ({self.enforce_p_vent}).")

      # Iterative procedure for scaling traction function to match vent pressure
      # If convex in the right way, the sequence of iterates are monotone
      for i in range(p_vent_max_num_iterations):
        def hydrostatic_RHS(x, p):
          yWt = self.yWt_fn(x)
          yC = self.yC_fn(x)
          T = self.T_fn(x)
          yWv = self.mixture.y_wv_eq(p, yWt, yC)
          return scale_value * self.traction_fn(x) - 9.8 / self.v_mix(p, T, yWv)
        soln = scipy.integrate.solve_ivp(hydrostatic_RHS,
            (self.x_mesh[0],self.x_mesh[-1]),
            np.array([self.p_chamber]),
            # t_eval=self.x_mesh,
            # method="Radau",
            max_step=0.5, # Likely mesh size in Quail
            dense_output=True,)
        p_vent_iterate = soln.y[0,-1]
        print(p_vent_iterate)
        scale_value += (p_vent_iterate - self.enforce_p_vent) / delta_p
        if np.abs(p_vent_iterate - self.enforce_p_vent) < p_vent_atol:
          break

    # Extract interpolator from scipy.integrate.solve_ivp
    dense_soln = soln.sol
    # Evaluate solution using interpolator
    Q = dense_soln(np.unique(x))
    # Check bounds of soln
    if not is_solve_direction_downward:
      if(soln.t[-1] < self.x_mesh[-1]):
        print("Warning: IVP terminated below top of domain.")
      # Extrapolate out-of-bounds values using nearest value
      last_legit_index = len(np.unique(x)) \
        - np.argmax(np.unique(x)[::-1] <= soln.t.max()) - 1
      Q[:, np.unique(x) > soln.t.max()] = \
        Q[:, last_legit_index:last_legit_index+1]

    # Compute solution in requested format
    if "p".casefold() == io_format.casefold() \
       or "native".casefold() == io_format.casefold():
      # Return solution state (p)
       # Evaluate solution using interpolator
      _output = Q
    elif "quail".casefold() == io_format.casefold():
      p = Q
      # Evaluate specified mass fractions, temperature
      yC = self.yC_fn(np.unique(x))
      yF = self.yF_fn(np.unique(x))
      yWt = self.yWt_fn(np.unique(x))
      yWv = self.mixture.y_wv_eq(p, yWt, yC)
      T = self.T_fn(np.unique(x))
      # Compute mixture intermediates
      v = self.v_mix(p, T, yWv)
      rho = 1.0 / v
      
      # Load and return conservative state vector
      U = np.zeros((*np.unique(x).shape, self.num_state_variables))
      U[...,0] = self.yA * rho
      U[...,1] = yWv * rho
      U[...,2] = (1.0 - (yWv + self.yA)) * rho
      U[...,3] = 0.0
      U[...,4] = (U[...,0] * self.mixture.air.c_v
                  + U[...,1] * self.mixture.waterEx.c_v
                  + U[...,2] * self.c_v_magma
                 ) * T
      U[...,5] = yWt * rho
      U[...,6] = yC * rho
      U[...,7] = yF * rho

      ''' Extract only values of U that correspond to query locations x. '''
      # Define associative map from value of x to state vector U
      vals = {x: U[i,:] for i, x in enumerate(np.unique(x))}
      U_out = np.zeros((*x.shape[:2], self.num_state_variables))
      # Map sample locations to state values U (possibly duplicated x)
      for i in range(U_out.shape[0]):
        for j in range(U_out.shape[1]):
          U_out[i,j,:] = vals[x[i,j,0]]
      _output = U_out
    else:
      raise ValueError(f"Unknown output format string '{io_format}'.")
    
    # Clean up and return output
    self.traction_fn = None
    self.yWt_fn = None
    self.yC_fn = None
    self.yF_fn = None
    self.T_fn = None
    self._has_lambda = False
    return _output

def parallel_forward_map(f:SteadyState, mg_p, mg_j0, num_processes=None):
  ''' Runs the forward ODE solution map in parallel.
  Specifying num_processes is highly preferred (otherwise the number of physical
  cores is checked; this may not work suitably for cluster usage.)
  '''
  import multiprocessing as mp

  if num_processes is None:
    # Estimate number of CPUs available
    try:
      from psutil import cpu_count
      # Saturate physical CPU count
      num_processes = cpu_count(logical=False)
    except ModuleNotFoundError:
      # Set default CPU count
      num_processes = 4
    
  # Define target function and argument list
  _args = list(zip(mg_p.ravel(), mg_j0.ravel()))
  with mp.Pool(num_processes) as pool:
    results = [r for r in pool.starmap(f.solve_ssIVP, _args, chunksize=4)]

  # Unpack results
  mg_x_top = np.zeros_like(mg_p)
  mg_p_top = np.zeros_like(mg_p)
  mg_T_in = np.zeros_like(mg_p)
  mg_T_top = np.zeros_like(mg_p)
  mg_v_in = np.zeros_like(mg_p)
  mg_v_top = np.zeros_like(mg_p)
  mg_M_top = np.zeros_like(mg_p)
  # Iteratively extract results
  for i in range(mg_p.ravel().size):
    _z = results[i] 
    mg_x_top.ravel()[i] = _z[0][-1]
    mg_p_top.ravel()[i] = _z[1][0,-1]
    T_in = f.T_ph(_z[1][0,0], _z[1][1,0], _z[1][2,0])
    mg_T_in.ravel()[i] = T_in
    T_top = f.T_ph(_z[1][0,-1], _z[1][1,-1], _z[1][2,-1])
    mg_T_top.ravel()[i] = T_top
    mg_v_in.ravel()[i] = mg_j0.ravel()[i] * f.v_mix(_z[1][0,0], T_in, _z[1][2,0])
    mg_v_top.ravel()[i] = mg_j0.ravel()[i] * f.v_mix(_z[1][0,-1], T_top, _z[1][2,-1])
    mg_M_top.ravel()[i] = mg_v_top.ravel()[i] / f.mixture.sound_speed(
      _z[1][0,-1], T_top, f.yA, _z[1][2,-1], 1.0 - f.yA - _z[1][2,-1])

  return {
    "x_top": mg_x_top,
    "p_top": mg_p_top,
    "T_in": mg_T_in,
    "T_top": mg_T_top,
    "vel_in": mg_v_in, 
    "vel_top": mg_v_top, 
    "M_top": mg_M_top,
    "results": results,
  }


if __name__ == "__main__":
  ''' Perform unit test '''

  ss = SteadyState(np.linspace(-3000,0,200))
  p_range = np.linspace(1e5, 10e6, 50)
  results_varp = [ss.solve_ssIVP(p_chamber=p_chamber, j0=2700*1.0) for p_chamber in p_range]
  print(results_varp[0])
  for result in results_varp:
    plt.plot(result[0], result[1][0,:], '.-')
 
  p_range = np.linspace(1e5, 10e6, 20)
  results_varp = [ss.solve_steady_state_problem(p_vent, 1.0, "u") for p_vent in p_range]
  u_range = np.linspace(0.01, 10, 10)
  results_varu = [ss.solve_steady_state_problem(1e5, u, "u") for u in u_range]

  x, (p, h, y) = ss.solve_steady_state_problem(0.5e5, 1.0, "u")
  plt.plot(x, p, '--', color="black")
  plt.show()

  ''' Plot sample solution '''
  x, (p, h, y) = ss.solve_steady_state_problem(1e5, 1.0, "u")
  plt.figure()
  plt.subplot(1,4,1)
  plt.plot(x, p, '.-')
  plt.subplot(1,4,2)
  plt.plot(x, h, '.-')
  plt.subplot(1,4,3)
  plt.plot(x, y, '.-')
  plt.subplot(1,4,4)
  phi = ss.mixture.vf_g(p, ss.mixture.T_ph(p, h, ss.yA, y, 1.0-ss.yA-y), ss.yA, y, 1.0-ss.yA-y)
  plt.plot(x, phi, '.-')
  plt.show()