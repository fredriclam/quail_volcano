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
	"File" : "../meshes/tungurahuaA7.msh",
}

Output = {
	"Prefix" : "tung5_atm7",
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

extend_atm = False

BoundaryConditions = {
	"ground7" : {
		"BCType" : "SlipWall",
	},
	"symmetry7" : {
		"BCType" : "SlipWall",
	},
	"r7" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r7",
	},
	"r6" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r6",
	},
}
LinkedSolvers = [
	{
		"DeckName": "r7r8.py",
		"BoundaryName": "r7",
	},
]

if not extend_atm:
	BoundaryConditions["r7"] = {
		"BCType" : "LinearizedImpedance2D",
	}
	LinkedSolvers = []