import numpy as np
from physics.multiphasevpT.hydrostatic1D import GlobalDG

REFINEMENT_FACTOR = 1.0

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 2.0,
	"NumTimeSteps" : 40000 * REFINEMENT_FACTOR,
  # TimeStepper options:
  # FE, SSPRK3, RK4, Strang (split for implicit source treatment)
	"TimeStepper" : "FE",
}

Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    # "ApplyLimiters" : ["WENO", "PositivityPreservingMultiphasevpT"],
    # "ApplyLimiters" : ["WENO"],
    # "ShockIndicator": "MinMod", "TVBParameter": 0.2,
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
        # Flag to use artificial viscosity
		# If true, artificial visocity will be added
    "ArtificialViscosity" : True,
	"AVParameter" : 500,#500,# 150, # 150 ~ 500 is ok for this commit #150, #50, #1e-5, #1e3, 5e3,
    'L2InitialCondition': False, # False <-> Use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "oscillationtest_conduit1_BCIC_p0",
  # Write to disk every WriteInterval timesteps
	"WriteInterval" : 400 * REFINEMENT_FACTOR,
	"WriteInitialSolution" : True,
  # Automatically queues up post_process.py after this file (see Quail examples)
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    # Use even number if using initial condition with discontinuous pressure
    "NumElemsX" : 3000, 
    "xmin" : -6000.0,
    "xmax" : 0.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

# Location of jump (sync both initial condition to specify jump in fluid
# composition, and in hydrostatic_solve to specify jump in pressure)
# x_jump = -550.0
# x_jump = -1500.0

# InitialCondition = {
# 	# "Function" : "UniformExsolutionTest",
#     "Function" : "RiemannProblem",
#     "xd": x_jump,
#     # "rhoL": 12.5,
#     # "uL": 0.0,
#     # "pL": 10*1e5,
#     # "rhoR": 1.25,
#     # "uR": 0.0,
#     # "pR": 1e5,
#     # arhoAL=1., arhoWvL=1., arhoML=2e5, uL=0., TL=1000., 
# 	            #  arhoAR=10., arhoWvR=0., arhoMR=0.125, uR=0., TR=300., xd=0.)
#     "arhoAL": 1e-3, # 4wt%, 5 MPa water design
#     "arhoWvL": 9.474849008229967e+00,
#     "arhoML": 3.142648542057735e+02, # Near critical, plenty of initial exsolved
# 	"TL": 1000.,
#     "arhoWtL": 1.294958812856014e+01,
# }

# InitialCondition = {
# 	# Initial condition (not necessarily hydrostatic; the injected function (see
#   # below) takes this initial condition and computes the hydrostatic solution).
#   # The parameters can be provided below, or changed in the source:
#   # quail_volcano\src\physics\multiphasevpT\functions.py > RiemannProblem
#   "Function" : "RiemannProblem",
#   # The following are optional parameters. If not provided, the default args
#   # are used.
#   # Left side values
#   "arhoAL": 1e-1,
#   "arhoWvL": 8.686,
#   "arhoML": 2600.,
#   "uL": 0.,
#   "TL": 1000.,
#   "arhoWtL": 75.0,
#   "arhoCL": 1.05e3, 
#   # Right side values
#   "arhoAR": 1.161,
#   "arhoWvR": 1.161*5e-3,
#   "arhoMR": 1e-6,
#   "uR": 0.,
#   "TR": 300.,
#   "arhoWtR": 1.161*5e-3,
#   "arhoCR": 1e-6,
#   "xd": x_jump, # Position of the discontinuity
# }

InitialCondition = {
    "Function": "SteadyState",
    "yC": 0.01,
    "yWt": 0.03,
    "yA": 1e-7,
    "yWvInletMin": 1e-7,
    "yCMin": 1e-7,
    "crit_volfrac": 0.7,
    "tau_d": 10.0,
    "tau_f": 10.0,
    "conduit_radius": 50,
    "conduit_length": Mesh["xmax"] - Mesh["xmin"],
    "T_chamber": 800+273.15,
    "c_v_magma": 3e3,
    "rho0_magma": 2.7e3,
    "K_magma": 10e9,
    "p0_magma": 5e6,
    "solubility_k": 5e-6,
    "solubility_n": 0.5,
    "NumElems": Mesh["NumElemsX"],
}

# # List of functions to inject in custom user function
# def hydrostatic_solve(solver, owner_domain=None):
#     GlobalDG(solver).set_initial_condition(
#         p_bdry=1e5,
#         is_jump_included=True,
#         owner_domain=owner_domain,
#         constr_key="YEq",
#         x_jump=x_jump,
#         # p_jump_factor=0.17054936309292384,
#     )

# Inject = [
#     # {
#     #     "Function": hydrostatic_solve,
#     #     "Initial": True,
#     #     "Postinitial": False,
#     # }
# ]

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
        "source_treatment" : "Explicit",
	},
    "source2": {
        "Function": "FrictionVolFracVariableMu",
        # "Function": "FrictionVolFracConstMu",
        "source_treatment" : "Explicit",
        # "mu": 1e5,
        # "crit_volfrac": 0.8,
    },
    "source3": {
        "Function": "ExsolutionSource", # likely problematic: check again after adjusting target y function?
        "source_treatment" : "Explicit",
        "tau_d": 10.0,
    },
    "source4": {
        "Function": "FragmentationTimescaleSource",
        "source_treatment": "Explicit",
        "crit_volfrac": 0.7,
        "tau_f": 10.0,
    }
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

extend_conduit = False # TODO: change back
if extend_conduit:
    BoundaryConditions = {
        "x1" : {
            "BCType" : "MultiphasevpT1D1D",
            "bkey": "interface_-1",
        },
        "x2" : { 
            "BCType" : "MultiphasevpT2D1D", # TODO: implement r-weighted integration
            "bkey": "vent",
        },
    }

    LinkedSolvers = [
        {
            "DeckName": "conduit2.py",
            "BoundaryName": "interface_-1",
        },
    ]
else:
    BoundaryConditions = {
        # "x1" : {
        #     "BCType" : "SlipWall",
        # },

        # The leftmost boundary
        "x1" : {
            # To be replaced by an exit pressure boundary condition
            #"BCType" : "SlipWall"
            "BCType" : "MassFluxInlet1D",
            "mass_flux" : 2700,
            "p_chamber" : 2e8,
            "T_chamber" : 800+273.15,
            # To use multiple domains (for parallelism), the below can be uncommented
            # and bkey set to a name that is known to this solver and a linked solver.
            # See LinkedSolvers below for parallelism
            # "BCType" : "MultiphasevpT1D1D",
            # "bkey": "interface_-1",
        },
        # "x2" : { 
        #     "BCType" : "MultiphasevpT2D1D",
        #     "bkey" : "vent",
        # },
        # "x2" : { 
        #     "BCType" : "SlipWall", # TODO: implement r-weighted integration (check)
        #     # "bkey": "vent",
        # },
        "x2" : { 
            "BCType" : "PressureOutlet1D", # TODO: implement r-weighted integration (check)
            "p": 1e5,
        },
        # "x2" : { 
        #     "BCType" : "MultiphasevpT2D1D", # TODO: implement r-weighted integration
        #     "bkey": "vent",
        # },
    }