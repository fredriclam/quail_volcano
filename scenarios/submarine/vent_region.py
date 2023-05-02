import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 5,
	"NumTimeSteps" : 5*2500,
	"TimeStepper" : "FE",
}

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : False,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	# "AVParameter" : 150,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/tungurahuaA1.msh",
}

Output = {
	"Prefix" : "submarine_proto_hydro1_atm1",
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

BoundaryConditions = {
	"r1" : {
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
		"BCType" : "LinearizedImpedance2D",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

LinkedSolvers = []
LinkedSolvers = [
# 	{
# 		"DeckName": "conduit.py",
# 		"BoundaryName": "vent",
# 	},
	{
		"DeckName": "r1r2.py",
		"BoundaryName": "r1",
	},
]