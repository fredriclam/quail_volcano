import numpy as np

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
	"AVParameter" : 0.3,#.30,#0.3,#5e3
	# "AVParameter" : 500,#5e3
	# 'L2InitialCondition': True, # Use interpolation instead of L2 projection of Riemann data
	'L2InitialCondition': False,
}

Mesh = {
	"File" : "../meshes/submarinetestA2.msh",
}

Output = {
	"Prefix" : "submarine_proto_WLMA11_atm2",
	"WriteInterval" : 5,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphaseWLMA",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "num_parallel_workers": 7,
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
	# "source4": {
	# 	"Function" : "CylindricalGeometricSource",
	# 	# "source_treatment" : "Explicit",
	# }
}

# Restart = {
# 	"File" : "___.pkl",
# 	"StartFromFileTime" : True
# }

InitialCondition = {
	"Function" : "IsothermalAtmosphere",
}

ExactSolution = InitialCondition.copy()

extend_atm = False

BoundaryConditions = {
	"ground2" : {
		"BCType" : "SlipWall",
	},
	"symmetry2" : {
		"BCType" : "SlipWall",
	},
	"r2" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r2",
	},
	"r1" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r1",
	},
}
LinkedSolvers = [
	{
		"DeckName": "r2r3.py",
		"BoundaryName": "r2",
	},
]

if not extend_atm:
	BoundaryConditions["r2"] = {
		"BCType" : "SlipWall",
	}
	LinkedSolvers = []