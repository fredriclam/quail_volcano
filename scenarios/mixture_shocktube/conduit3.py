import numpy as np
import copy

# TimeStepping is not included in linked solvers

Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    # "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO"], "ShockIndicator": "MinMod",
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
        # Flag to use artificial viscosity
		# If true, artificial visocity will be added
    "ArtificialViscosity" : False,
	"AVParameter" : 100, #150, #50, #1e-5, #1e3, 5e3,
    'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "mixture_shocktube_conduit3",
	"WriteInterval" : 4*200,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 30*301, #151,#351,
    "xmin" : 600.0,
    "xmax" : 1800.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
	"Function" : "RiemannProblem",
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
  "x1" : {
      "BCType" : "MultiphasevpT1D1D", # Link
      "bkey": "xupper",
  },
  "x2" : { 
      "BCType" : "MultiphasevpT1D1D",
      "bkey": "xupperupper",
  },
}

LinkedSolvers = [
	{
		"DeckName": "conduit5.py",
		"BoundaryName": "xupperupper",
	},
]