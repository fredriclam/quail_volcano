import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 0.002,
	"NumTimeSteps" : 10000,
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

Mesh = {
	"File" : "../meshes/rectangle.msh", # Unit square
}

Output = {
	"Prefix" : "noh_mixture_nondilute",
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
	"mode": "nondilute",
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "SlipWall",
	},
	"x2" : {
		"BCType" : "NohInletMixture",
	},
	"y1" : {
		"BCType" : "SlipWall",
	},
	"y2" : {
		"BCType" : "SlipWall",
	},
}