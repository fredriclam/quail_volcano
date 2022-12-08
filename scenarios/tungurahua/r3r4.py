import numpy as np

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 150,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/tungurahuaA4.msh",
}

Output = {
	"Prefix" : "tung3_atm4",
	"WriteInterval" : 2000,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
	},
	# "source3": {
	# 		"Function": "ExsolutionSource",
	# 		"source_treatment" : "Implicit",
	# },
	"source4": {
		"Function" : "CylindricalGeometricSource",
		# "source_treatment" : "Explicit",
	}
}

# Restart = {
# 	"File" : "___.pkl",
# 	"StartFromFileTime" : True
# }

InitialCondition = {
	"Function" : "LinearAtmosphere",
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"ground4" : {
		"BCType" : "SlipWall",
	},
	"symmetry4" : {
		"BCType" : "SlipWall",
	},
	"r4" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r4",
	},
	"r3" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r3",
	},
}
LinkedSolvers = [
	{
		"DeckName": "r4r5.py",
		"BoundaryName": "r4",
	},
]