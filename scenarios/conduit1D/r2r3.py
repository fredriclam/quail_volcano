import numpy as np

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.0,
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 200,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/volcanoA3.msh",
}

Output = {
	"Prefix" : "atm3SteadyState",
	"WriteInterval" : 200,
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
		# "source_treatment" : "Explicit",
	},
}

# Restart = {
# 	"File" : "atm3SteadyState_inlet_140.pkl",
# 	"StartFromFileTime" : True
# }

InitialCondition = {
	"Function" : "LinearAtmosphere",
	# "state" : UQuiescent,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"ground3" : {
		"BCType" : "SlipWall",
	},
	"symmetry3" : {
		"BCType" : "SlipWall",
	},
	# "r3" : {
	# 	"BCType" : "Euler2D2D",
	# 	"bkey": "r3",
	# },
	"r3" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r3",
	},
	"r2" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r2",
	},
}

LinkedSolvers = []
LinkedSolvers = [
	#{
	#	"DeckName": "conduit.py",
	#	"BoundaryName": "vent",
	#},
	{
		"DeckName": "r3r4.py",
		"BoundaryName": "r3",
	},
]
