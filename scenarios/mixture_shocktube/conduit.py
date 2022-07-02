import numpy as np
import copy

# Brute force P0 setting
use_P0_detailed = False
if use_P0_detailed:
    TimeStepping = {
        "InitialTime" : 0.0,
        "FinalTime" : 0.5*2*2.0, #0.1 @ meter scale
        "NumTimeSteps" : 2*10*20000,#2*20000*4,#5000*2, #13000*2, #5000 @ meter scale
    # 100000 for generalB1, 400~K
        "TimeStepper" : "FE",
    }
else:
    TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 0.050, #0.1 @ meter scale
	"NumTimeSteps" : 500, # 20000,#2*20000*4,#5000*2, #13000*2, #5000 @ meter scale
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
	"AVParameter" : 100, # 150 ~ 500 is ok for this commit #150, #50, #1e-5, #1e3, 5e3,
    'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "mixture_shocktube_conduit",
	"WriteInterval" : 10,#4*200,
	"WriteInitialSolution" : True,
	"AutoPostProcess": True,
}

if use_P0_detailed:
    Mesh = {
        "File" : None,
        "ElementShape" : "Segment",
        "NumElemsX" : 30*301, #151,#351,
        "xmin" : -600.0,
        "xmax" : 600.0,
    }
else: 
    Mesh = {
        "File" : None,
        "ElementShape" : "Segment",
        "NumElemsX" : 3*301, #151,#351,
        "xmin" : -600.0,
        "xmax" : 600.0,
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
	"Function" : "RiemannProblem",
    # "rhoL": 12.5,
    # "uL": 0.0,
    # "pL": 10*1e5,
    # "rhoR": 1.25,
    # "uR": 0.0,
    # "pR": 1e5,
    # arhoAL=1., arhoWvL=1., arhoML=2e5, uL=0., TL=1000., 
	            #  arhoAR=10., arhoWvR=0., arhoMR=0.125, uR=0., TR=300., xd=0.)
}

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
        "source_treatment" : "Explicit",
	},
    "source2": {
        "Function": "FrictionVolFracConstMu",
        "source_treatment" : "Explicit",
    },
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

IsDecoupled = True
if IsDecoupled:
    BoundaryConditions = {
        "x1" : {
            "BCType" : "SlipWall"
        },
        "x2" : { 
            "BCType" : "SlipWall",
        },
    }
else:
    BoundaryConditions = {
        "x1" : {
            "BCType" : "MultiphasevpT1D1D",
            "bkey": "xlower",
        },
        "x2" : { 
            #   "BCType" : "SlipWall",
            "BCType" : "MultiphasevpT1D1D",
            "bkey": "xupper",
        },
    }
    LinkedSolvers = [
	{
		"DeckName": "conduit2.py",
		"BoundaryName": "xlower",
	},
    {
		"DeckName": "conduit3.py",
		"BoundaryName": "xupper",
	},
]