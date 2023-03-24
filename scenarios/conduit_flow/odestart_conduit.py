import numpy as np

# Set timestepper
TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 360,
	# "NumTimeSteps" : 30*8000, # 4000 (@ dx = 2.0)
	"TimeStepper" : "RK3SR",
	"NumTimeSteps" : 360*7000, # 4000 (@ dx = 2.0)
  # If CFL-limited: LS-SSPRK3-P2 as implemented:
  # For sound speed <= 1925 m/s
  #     CFL ~ (1925+u) * (2k+1) / dx is generous for RK3 (cf. 2k+1 --> 1/.209)
  # Values for NumTimeSteps * dx for different strategies:
  #   Plugin RK3 (k = 2) for u = 0:
  #     9625
  #   Linear advection numerical estimate (Cockburn & Shu 2001):
  #     (1925+u) / .209 >= 9210.5
  #   Empirical greed, SSPRK3, dx = 1:
  #     7000
  # The SSPRK3 implementation has CFL coefficient > 1
  # Numerical explosion happens under the exsolution front
  #
  # If CFL-limited: LS-RK3-P2: (custom RK3)
  #   Empirical greed: 10000 (85.7% cost of SSPRK3)
  #   At 10000, (CFL_coeff=1) * (dx=1m) / (2p+1=5) / (dt=1e-4) = 2000 m/s
  #   Resolvable 2000 m/s cf. 1925 sound speed => safety factor ~ 1.04 required
  #
  # If CFL-limited: LS-RK3SR-P2 (optimal CFL efficiency):
  #   Empirical greed: 6500 (74.3% cost of SSPRK3)
  #   At 6500, (CFL_coeff=2) * (dx=1m) / (2p+1=5) / (dt=1.53846e-4) = 2600 m/s
  #   Resolvable 2600 m/s cf. 1925 sound speed => safety factor ~ 1.35 required
  #
  # FE may be source-term-limited (oscillatory sources outside A-stability region)
  # and Cockburn & Shu 2001 indicate that the method may be unstable at
  # constant dt/dx.
}

Numerics = {
  # Solution order; these correspond to:
  # 0: 1 node in each element representing the average value
  # 1: 2 nodes in each element constituting a linear representation of the
  #    solution in the element
  # 2: 3 nodes in each element constituting a quadratic representation of the
  #    solution in the element
  "SolutionOrder" : 2,
  "SolutionBasis" : "LagrangeSeg",
  "Solver" : "DG",
  "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
  "ElementQuadrature" : "GaussLegendre",
  "FaceQuadrature" : "GaussLegendre",
  # Artificial viscosity adds a diffusion term to all equations, where the
  # strong form residual is large and where the pressure gradient relative to
  # hydrostatic is large
  "ArtificialViscosity" : True,
  "AVParameter" : 0.3,
  # If L2InitialCondition is false, use interpolation instead of L2 projection of Riemann data
  'L2InitialCondition': True,
}

Output = {
	"Prefix" : "/scratch/users/kcoppess/ODEsteadyState/conduit",
	#"Prefix" : "injections/conduit",
  # Write to disk every WriteInterval timesteps
	"WriteInterval" : 800,
	"WriteInitialSolution" : True,
  # Automatically queues up post_process.py after this file (see Quail examples)
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    # Use even number if using initial condition with discontinuous pressure
    "NumElemsX" : 3000,
    "xmin" : -3000 - 150.0, # -6000.0 - 150.0,
    "xmax" : 0.0 - 150.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}


''' Initial condition stuff '''
# Mass fractions at t = 0
phi_crys = 0.4025 * (1.1 - 0.1 * np.sin(0.0))
chi_water = 0.05055
yWt_init = chi_water * (1 - phi_crys) / (1 + chi_water)
yC_init = phi_crys
# Compute representation of the 1D mesh. This part is overriden if
# generate_conduit_partitions is used.
n_elems_global = Mesh["NumElemsX"]
if Numerics["SolutionOrder"] == 0:
    n_nodes_global = n_elems_global
elif Numerics["SolutionOrder"] == 1:
    n_nodes_global = n_elems_global + 1
elif Numerics["SolutionOrder"] == 2:
    n_nodes_global = 2*n_elems_global + 1
else:
    raise ValueError("Oops, is there solution order > 2?")
x_global = np.linspace(Mesh["xmin"], Mesh["xmax"], n_nodes_global)

InitialCondition = {
    "Function": "SteadyState",
    "x_global": x_global,
    "p_vent": 1e5,          # Vent pressure
    "inlet_input_val": 1.0, # Inlet velocity; see also BoundaryCondition["x1"]
    "input_type": "u",
    "yC": lambda t: 0.4025 * (1.1 - 0.1 * np.sin(2*np.pi*t/4.0)), # yC_init,
    "yWt": lambda t: 0.05055 / (1.0 + 0.05055) \
      * (1.0 - 0.4025 * (1.1 - 0.1 * np.sin(2*np.pi*0/4.0))), # yWt_init, !!! t = 0, frozen
    "yA": 1e-8,
    "yWvInletMin": 1e-8,
    "yCMin": 1e-8,
    "crit_volfrac": 0.8,
    "tau_d": 1.0,
    "tau_f": 3.0, # Resolvable in space( tau_f >~ dx/u)? when frag close to bndry
    "conduit_radius": 50,
    "T_chamber": 1000,
    "c_v_magma": 3e3,
    "rho0_magma": 2.7e3,
    "K_magma": 10e9,
    "p0_magma": 5e6,
    "solubility_k": 5e-6,
    "solubility_n": 0.5,
    "approx_massfracs": True,
}

# Add source terms here. Source terms just stack up, and can be named whatever
# is convenient. source_treatment is only relevant for splitting schemes, like
# "Strang" in TimeStepping options.
SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
    "source_treatment" : "Explicit",
	},
  "source2": {
      "Function": "FrictionVolFracVariableMu",
      "source_treatment" : "Explicit",
      # Some options, and their default values
      "conduit_radius": InitialCondition["conduit_radius"],
  },
  "source3": {
      "Function": "ExsolutionSource",
      "source_treatment" : "Explicit",
      "tau_d": InitialCondition["tau_d"],
  },
  "source4": {
      "Function": "FragmentationTimescaleSource",
      "source_treatment" : "Explicit",
      "tau_f": InitialCondition["tau_f"],
      "crit_volfrac": InitialCondition["crit_volfrac"],
  },
}

# Fake exact solution (use something cheap)
ExactSolution = {
  "Function": "RiemannProblem",
}

BoundaryConditions = {
    # The leftmost boundary
    "x1" : {
      "BCType" : "VelocityInlet1D",
      "u" : InitialCondition["inlet_input_val"],
      "p_chamber" : 100e6,
      "T_chamber" : InitialCondition["T_chamber"],
      "trace_arho": 1e-8*2700,
    },
    "x2": {
      "BCType" : "PressureOutlet1D",
      "p": 1e5,
    },
    # "x2" : { 
    #   "BCType" : "MultiphasevpT2D1D",
    #   "bkey" : "vent",
    # },
}

# The solvers/domains that are linked to this one through a coupling BC.
# DeckName refers to the name of a copy of this file, specifying the parameters
# for the solver on the linked domain. The linked domain does not need to link
# back to this one, and does not need a TimeStepping option (TimeStepping
# options are taken from this root parameter file).
# The BoundaryName is the name given to the coupling boundary condition. It
# should be unique across all linked solvers, and named in the bkey of the
# corresponding BoundaryCondition (for example, for boundary "x1" here and 
# boundary "x2" in the linked parameter file).

# LinkedSolvers = [
#     {
#         "DeckName": "odestart_vent_region.py",
#         "BoundaryName": "vent",
#     },
# ]
LinkedSolvers = []
