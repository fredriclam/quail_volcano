import numpy as np
import copy
from physics.multiphasevpT.hydrostatic1D import GlobalDG


# TimeStepping is not included in linked solvers

Numerics = {
    "SolutionOrder" : 2,
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
    "ArtificialViscosity" : True,
	"AVParameter" : 150, # 150 ~ 500 is ok for this commit #150, #50, #1e-5, #1e3, 5e3,
    'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "mixture_shocktube_conduit2_trash",
	"WriteInterval" : 30,#4*200,
	"WriteInitialSolution" : True,
	"AutoPostProcess": True,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 150,
    "xmin" : -1800.0,
    "xmax" : -600.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
	"Function" : "RiemannProblem",
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
    "source3": {
        "Function": "ExsolutionSource",
        "source_treatment" : "Implicit",
    },
}

# List of functions to inject in custom user function
def hydrostatic_solve(solver, owner_domain=None):
    GlobalDG(solver).set_initial_condition(
        p_bdry=None,
        is_jump_included=False,
        owner_domain=owner_domain,
        # traction_fn=lambda x: (-1e5)*np.exp(-((x-0.0)/50.0)**2.0)
    )

Inject = [
    {
        "Function": hydrostatic_solve,
        "Initial": True,
        "Postinitial": False,
    }
]

# Fake exact solution
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
  "x1" : {
      "BCType" : "SlipWall",
    #   "BCType" : "MultiphasevpT1D1D",
    #   "bkey": "xlowerlower",
  },
  "x2" : { 
      "BCType" : "MultiphasevpT1D1D", # Link
      "bkey": "interface_-1",
  },
}

# LinkedSolvers = [
# 	{
# 		"DeckName": "conduit4.py",
# 		"BoundaryName": "xlowerlower",
# 	},
# ]