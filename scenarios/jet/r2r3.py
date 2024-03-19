import numpy as np

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 30,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/conical1_3.msh",
}

Output = {
	"Prefix" : "jet_atm3_test2",
	"WriteInterval" : 100,
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
	"Function" : "IsothermalAtmosphere",
  "h0": -1100, # TODO: fix
  "massFracM": 1e-5,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"ground3" : {
		"BCType" : "SlipWall",
	},
	"symmetry3" : {
		"BCType" : "SlipWall",
	},
	"r3" : {
		# "BCType" : "MultiphasevpT2D2D",
		"BCType" : "SlipWall",
		# "bkey": "r3",
	},
	"r2" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r2",
	},
}
# LinkedSolvers = [
# 	{
# 		"DeckName": "r3r4.py",
# 		"BoundaryName": "r3",
# 	},
# ]