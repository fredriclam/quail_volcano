import numpy as np
from physics.multiphasevpT.hydrostatic1D import GlobalDG

## Set timestepper
#TimeStepping = {
#	"InitialTime" : 0.0,
#	"FinalTime" : 5.0,
#	"NumTimeSteps" : 25000,
#  # TimeStepper options:
#  # FE, SSPRK3, RK4, Strang (split for implicit source treatment)
#	"TimeStepper" : "FE",
#}

Numerics = {
  # Solution order; these correspond to:
  # 0: 1 node in each element representing the average value
  # 1: 2 nodes in each element constituting a linear representation of the
  #    solution in the element
  # 2: 3 nodes in each element constituting a quadratic representation of the
  #    solution in the element
  # Order 0 is most robust, and the error is mostly diffusive (may be
  # currently bugged, as of commit 68ce3b653480e52a1d52436f0206b3327c8f744f
  # on branch main.
  "SolutionOrder" : 1,
  "SolutionBasis" : "LagrangeSeg",
  "Solver" : "DG",
  "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
  # The following is a way to stack additional limiters (WENO limiter is
  # designed for the Euler equations and reduces oscillations while preserving
  # order of accuracy)
  # "ApplyLimiters" : ["WENO", "PositivityPreservingMultiphasevpT"],
  #   "ShockIndicator": "MinMod", "TVBParameter": 0.2,
  "ElementQuadrature" : "GaussLegendre",
  "FaceQuadrature" : "GaussLegendre",
  # Artificial viscosity adds a diffusion term to all equations, proportional to
  #   |grad p| / p + |grad (alpha_a * rho_a)| / (alpha_a * rho_a),
  # the cube of the element size (h^3), and AVParameter.
  "ArtificialViscosity" : True,
  "AVParameter" : 500,
  'L2InitialCondition': False, # If false, use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "steadyState_cVF40/conduit1",
  # Write to disk every WriteInterval timesteps
	"WriteInterval" : 1000,
	"WriteInitialSolution" : True,
  # Automatically queues up post_process.py after this file (see Quail examples)
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    # Use even number if using initial condition with discontinuous pressure
    "NumElemsX" : 1000, 
    "xmin" : -3500.0,
    "xmax" : -2500.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
	# Initial condition (not necessarily hydrostatic; the injected function (see
  # below) takes this initial condition and computes the hydrostatic solution).
  # The parameters can be provided below, or changed in the source:
  # quail_volcano\src\physics\multiphasevpT\functions.py > RiemannProblem
  "Function" : "RiemannProblem",
  # The following are optional parameters. If not provided, the default args
  # are used.
  # Left side values
  "arhoAL": 1e-1,
  "arhoWvL": 8.686,
  "arhoML": 2600.,
  "uL": 0.,
  "TL": 1000.,
  "arhoWtL": 75.0,
  "arhoCL": 1.05e3, 
  # Right side values
  "arhoAR": 1.161,
  "arhoWvR": 1.161*5e-3,
  "arhoMR": 1e-6,
  "uR": 0.,
  "TR": 300.,
  "arhoWtR": 1.161*5e-3,
  "arhoCR": 1e-6,
  "xd": -400.0, # Position of the discontinuity
}

# Define the hydrostatic steady-state solver that operates on the initial
# condition provided. The solver solves the discontinuous Galerkin steady-state
# problem, and preserves certain properties of the provided initial condition.
# See hydrostatic1D.py for more details
def hydrostatic_solve(solver, owner_domain=None):
    GlobalDG(solver).set_initial_condition(
        p_bdry=None,
        is_jump_included=False,
        owner_domain=owner_domain,
        constr_key="YEq",
        # To set the traction function, use the following line and prescribe
        # traction as a function of x. The traction function needs to be
        # somehow compatible with the initial condition in RiemannProblem.
        # Maybe even a smoothed out version of Riemann problem would be needed.
        # traction_fn=lambda x: (-1e7/(50.0*np.sqrt(np.pi)))*np.exp(-((x-0.0)/50.0)**2.0)
    )

# Inject is a list of dicts with a value "Function" pointing to a function
# provided args (solver, (optional: owner_domain)). These functions are injected
# into Quail to make them run before the first timestep, and once after each
# timestep. If Initial is provided True, the function runs before the first timestep.
# If Postinitial is provided True, the function runs after each timestep.
Inject = [
    {
        "Function": hydrostatic_solve,
        "Initial": True,
        "Postinitial": False,
    }
]

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
      # "mu": 1e5,
      # "conduit_radius": 50.0,
			# "crit_volfrac": 0.8,
      # "logistic_scale": 0.01,
  },
  "source3": {
      "Function": "ExsolutionSource",
      "source_treatment" : "Implicit",
      "tau_d": 1.0,
  },
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    # The leftmost boundary
    "x1" : {
      # To be replaced by an exit pressure boundary condition
      #"BCType" : "SlipWall"
      #"BCType" : "MassFluxInlet1D",
      #"mass_flux" : 2700,
      #"p_chamber" : 2e8,
      #"T_chamber" : 1000,
      # To use multiple domains (for parallelism), the below can be uncommented
      # and bkey set to a name that is known to this solver and a linked solver.
      # See LinkedSolvers below for parallelism
      "BCType" : "MultiphasevpT1D1D",
      "bkey": "interface_-2",
    },
    "x2" : { 
        "BCType" : "MultiphasevpT1D1D",
        "bkey" : "interface_-1",
    },
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
LinkedSolvers = [
    {
        "DeckName": "steadyState_conduit2.py",
        "BoundaryName": "interface_-2",
    },
    #{
    #    "DeckName": "steadyState_conduit0.py",
    #    "BoundaryName": "interface_-1",
    #},
]
