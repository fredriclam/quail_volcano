import numpy as np

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 0.3, #5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/tungurahuaA1.msh",
}

Output = {
	"Prefix" : "tungurahua_StaticPlugtest_vent_sub0",
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
  "r1" : {
		"BCType" : "SlipWall",
	},
	# "r1" : {
	# 	"BCType" : "MultiphasevpT2D2D",
	# 	"bkey": "r1",
	# },
	"ground" : {
		"BCType" : "SlipWall",
	},
	"flare" : {
		"BCType" : "SlipWall",
	},
	"pipewall" : {
		"BCType" : "SlipWall",
	},
	"x2" : {
		"BCType" : "MultiphasevpT2D1D",
		"bkey": "comm2D1D",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

LinkedSolvers = []
# LinkedSolvers = [
# 	{
# 		"DeckName": "conduit.py",
# 		"BoundaryName": "vent",
# 	},
# 	{
# 		"DeckName": "r1r2.py",
# 		"BoundaryName": "r1",
# 	},
# ]