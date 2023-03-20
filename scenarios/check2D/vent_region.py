import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 1,
	"NumTimeSteps" : 4000,
	"TimeStepper" : "SSPRK3",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 0.3,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/volcanoA1.msh",
}

Output = {
	"Prefix" : "check2D_atm_2",
	"WriteInterval" : 400,
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
	"source4": {
		"Function" : "CylindricalGeometricSource",
	}
}

InitialCondition = {
	"Function" : "IsothermalAtmosphere", #TODO: use isothermal atmosphere
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"r1" : {
		# "BCType" : "SlipWall",
		# "BCType" : "MultiphasevpT2D2D",
		"BCType" : "LinearizedIsothermalOutflow2D",
		# "bkey": "r1",
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
		"BCType" : "SlipWall",
		# "BCType" : "MultiphasevpT2D1D",
		# "bkey": "vent",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

# LinkedSolvers = []
LinkedSolvers = [
	# {
	# 	"DeckName": "conduit.py",
	# 	"BoundaryName": "vent",
	# },
	# {
	# 	"DeckName": "r1r2.py",
	# 	"BoundaryName": "r1",
	# },
]