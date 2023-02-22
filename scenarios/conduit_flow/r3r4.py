import numpy as np

Restart = {
    #"File" : "steadyState_3m_sherlock_r4/atm4_519.pkl",
    "File" : "steadyState_smoothing/r1atm4_final.pkl",
    "StartFromFileTime" : False,
}

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
	"File" : "../meshes/volcanoA4.msh",
}

Output = {
	#"Prefix" : "steadyState_smoothing/r1atm4",
	"Prefix" : "injections/atm4",
	"WriteInterval" : 800,
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
	"ground4" : {
		"BCType" : "SlipWall",
	},
	"symmetry4" : {
		"BCType" : "SlipWall",
	},
	"r4" : {
		"BCType" : "LinearizedImpedance2D",
		# "bkey": "r4",
	},
	"r3" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r3",
	},
}

LinkedSolvers = []
