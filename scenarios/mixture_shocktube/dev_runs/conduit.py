import numpy as np
from physics.multiphasevpT.hydrostatic1D import GlobalDG

# Brute force P0 setting
use_P0_detailed = False
# if use_P0_detailed:
#     TimeStepping = {
#         "InitialTime" : 0.0,
#         "FinalTime" : 0.5*2*2.0, #0.1 @ meter scale
#         "NumTimeSteps" : 2*10*20000,#2*20000*4,#5000*2, #13000*2, #5000 @ meter scale
#     # 100000 for generalB1, 400~K
#         "TimeStepper" : "FE",
#     }
# else:
TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 1.0,#0.030,#1.0, #0.1 @ meter scale
	"NumTimeSteps" : 5000,#60,#2000,#1*1000, # 20000,#2*20000*4,#5000*2, #13000*2, #5000 @ meter scale
     # 100000 for generalB1, 400~K
	"TimeStepper" : "Strang",
}

Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    # "ApplyLimiters" : ["WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.2,
    # "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.0,
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
        # Flag to use artificial viscosity
		# If true, artificial visocity will be added
    "ArtificialViscosity" : False,
	"AVParameter" : 50, # 50 for P2, 60 elems over 1200 domain length at CFL~0.1, O(10Mpa) jump
    'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "v0_pboundary",
	"WriteInterval" : 1,#4*200,
	"WriteInitialSolution" : True,
	"AutoPostProcess": True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 600, # Use even number if using p_jump IC
    "xmin" : -1200.0 - 150.0,
    "xmax" : 0.0     - 150.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

# SourceTerms = {
#     "source1" : {
#         "Function" : "PorousSource",
#         "nu": 0.0*-.5,
#         "alpha": 0.0*2.0e4,
#         "T_m": 400.0,
#     },
# }

# pAmbient = 30*1e5
# TAmbient = 1000.0 # 300.0
# rhoAmbient = pAmbient / ( Physics["GasConstant"] * TAmbient )
# eAmbient = rhoAmbient * Physics["GasConstant"] / (Physics["SpecificHeatRatio"] - 1.0) * TAmbient
# rhoAmbient = 3.0 # 1.2
# pAmbient = rhoAmbient * Physics["GasConstant"] * TAmbient

if False:
    # Sod state
    rhoAmbient = 1.0 # 1.2
    # TAmbient = 400.0 # 300.0
    pAmbient = 1.0
    eAmbient = pAmbient / (Physics["SpecificHeatRatio"] - 1.0)

# UQuiescent = np.array([rhoAmbient, 0.0, eAmbient])

InitialCondition = {
	# "Function" : "UniformExsolutionTest",
    "Function" : "RiemannProblem",
    "xd": -600.0-150.0,
    "arhoWvL": 6,#8.686,
    "arhoML": 2400,#2496.3,
    "arhoWvR": 1006,#8.686,
    "arhoMR": 0,#2496.3,
    "arhoAL": 0,
    "arhoAR": 1.2,
    "TL": 1000,
    "TR": 300,
    # "rhoL": 12.5,
    # "uL": 0.0,
    # "pL": 10*1e5,
    # "rhoR": 1.25,
    # "uR": 0.0,
    # "pR": 1e5,
    # arhoAL=1., arhoWvL=1., arhoML=2e5, uL=0., TL=1000., 
	            #  arhoAR=10., arhoWvR=0., arhoMR=0.125, uR=0., TR=300., xd=0.)
}

# List of functions to inject in custom user function
def hydrostatic_solve(solver, owner_domain=None):
    GlobalDG(solver).set_initial_condition(
        p_bdry=1e7,
        is_jump_included=True,
        owner_domain=owner_domain,
        x_jump=-600.0-150.0,
        constr_key="YEq",
        # traction_fn=lambda x: (-1e7/(50.0*np.sqrt(np.pi)))*np.exp(-((x-0.0)/50.0)**2.0)
    )

Inject = [
    {
        "Function": hydrostatic_solve,
        "Initial": True,
        "Postinitial": False,
    }
]

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
        "source_treatment" : "Explicit",
	},
    # Important note: velocity jitters happen near the head of the expansion fan
    # More viscosity turns out to be better
    "source2": {
        "Function": "FrictionVolFracConstMu",
        "source_treatment" : "Explicit",
        "mu": 1e6,
        "conduit_radius": 50.0,
		"crit_volfrac": 0.7,
        "logistic_scale": 0.01,
    },
    # "source3": {
    #     "Function": "ExsolutionSource",
    #     "source_treatment" : "Implicit",
    #     "tau_d" : 1.0,
    # },
}

if False:
    InitialCondition = {
        "Function" : "MixtureRiemann1",
        "rhoL": 1.0,
        "uL": 0.0,
        "pL": 1.0,
        "rhoR": 0.125,
        "uR": 0.0,
        "pR": 0.1,
    }

# Fake exact solution
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
"x1" : {
    "BCType" : "Inlet"
    # "BCType" : "MultiphasevpT1D1D",
    # "bkey": "interface_-1",
},
"x2" : { 
    # "BCType" : "SlipWall",
    "BCType" : "PressureOutlet",
    "p": 1e5,
    # "BCType" : "MultiphasevpT2D1D",
    # "bkey": "vent",
},
}

# LinkedSolvers = [
# {
#     "DeckName": "conduit_depth2.py",
#     "BoundaryName": "interface_-1",
# },
# ]