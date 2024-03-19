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
	"File" : "../meshes/conical1_2.msh",
}

Output = {
	"Prefix" : "jet_atm2_test2",
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

extend_atm = True

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
		"BCType" : "LinearizedImpedance2D",
	}
	LinkedSolvers = []