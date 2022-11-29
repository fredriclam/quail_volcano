import numpy as np

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.0,
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
	"File" : "../meshes/volcanoA2.msh",
}

Output = {
	"Prefix" : "mixture_shocktube_atm2_cyl",
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
# 	"File" : "atm2SteadyState_inlet_640.pkl",
# 	"StartFromFileTime" : True
# }

#if False:
#	# Sod state
#	rhoAmbient = 0.125
#	pAmbient = 0.1
#	eAmbient = pAmbient / (Physics["SpecificHeatRatio"] - 1.0)

InitialCondition = {
	"Function" : "LinearAtmosphere",
	# "state" : UQuiescent,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"ground_far" : {
		"BCType" : "SlipWall",
	},
	"symmetry_far" : {
		"BCType" : "SlipWall",
	},
	"r2" : {
		# "BCType" : "MultiphasevpT2D2D",
		# "BCType" : "SlipWall",
		"BCType" : "LinearizedImpedance2D",
		# "bkey": "r2",
	},
	"r1" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r1",
	},
}

# LinkedSolvers = []
if False:
	LinkedSolvers = [
		#{
		#	"DeckName": "conduit.py",
		#	"BoundaryName": "vent",
		#},
		{
			"DeckName": "r2r3.py",
			"BoundaryName": "r2",
		},
	]