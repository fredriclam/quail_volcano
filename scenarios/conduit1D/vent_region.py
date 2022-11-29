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
	"File" : "../meshes/volcanoA1.msh",
}

Output = {
	"Prefix" : "atmSteadyState_inlet_cont",
	"WriteInterval" : 2*200,
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
	# 		"source_treatment" : "Explicit",
	# # 		"source_treatment" : "Implicit",
	# },
}

Restart = {
	"File" : "atmSteadyState_inlet_640.pkl",
	"StartFromFileTime" : True
}

InitialCondition = {
	"Function" : "LinearAtmosphere",
	# "state" : UQuiescent,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"r1" : {
		# "BCType" : "SlipWall",
		# "BCType": "LinearizedImpedance2D",
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r1",
	},
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
		# "BCType" : "SlipWall",
		"BCType" : "MultiphasevpT2D1D",
		"bkey": "vent",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

# LinkedSolvers = []
LinkedSolvers = [
	#{
	#	"DeckName": "conduit.py",
	#	"BoundaryName": "vent",
	#},
	{
		"DeckName": "r1r2.py",
		"BoundaryName": "r1",
	},
]