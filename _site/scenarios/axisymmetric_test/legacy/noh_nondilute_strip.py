import numpy as np

use_1D = True

if use_1D:
  TimeStepping = {
    "InitialTime" : 0.0,
    "FinalTime" : 55.22732596279061,
    "NumTimeSteps" : 40000,
    "TimeStepper" : "FE", #"RK3SR", # "FE",
  }
else:
  TimeStepping = {
    "InitialTime" : 0.0,
    "FinalTime" : 0.0001,
    "NumTimeSteps" : 40000,
    "TimeStepper" : "FE",
  }

Numerics = {
	"SolutionOrder" : 0, #AV10:1,# 2,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.0,
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True, #AV10:True, #True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 30, #noh_problem2:0.01,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

if use_1D:
  Mesh = {
    'ElementShape': 'Segment',
    'File': None,
    'NumElemsX': 2000,
    'xmax': 1,
    'xmin': 0,
	}
  Numerics["SolutionBasis"] = "LagrangeSeg"
  Numerics["ApplyLimiters"] = []
else:
  # Load unit square from file
  Mesh = {
    "File" : "../meshes/rectangle.msh", # Unit square
  }
  # Built-in periodic strip
  Mesh = {
    "ElementShape" : "Triangle",
    "NumElemsX" : 500,
    "NumElemsY" : 10,
    "xmin" : 0,
    "xmax" : 0.25,
    "ymin" : -0.0025,
    "ymax" : 0.0025,
    "PeriodicBoundariesY" : ["y2", "y1"],
  }

Output = {
	"Prefix" : "noh_mixture_dilute_1d",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

SourceTerms = {
	"source4": {
		"Function" : "CylindricalGeometricSource",
	}
}

InitialCondition = {
	"Function" : "NohProblemMixture",
	"mode": "dilute",
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "SlipWall",
    "use_stagnation_correction": False,
		"Q": 1e2, # Max density factor limiter
	},
	"x2" : {
		"BCType" : "NohInletMixture",
	},
}