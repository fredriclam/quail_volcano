import numpy as np

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : False,
	# "AVParameter" : 150,#5e3
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/tungurahuaA2.msh",
}

Output = {
	"Prefix" : "submarine_proto_hydro1_atm2",
	"WriteInterval" : 25,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphaseWLMA",
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