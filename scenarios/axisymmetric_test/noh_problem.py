import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 1.0,
	"NumTimeSteps" : 5000,
	"TimeStepper" : "FE",
}

Numerics = {
	"SolutionOrder" : 1,# 2,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.0,
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True, #True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 1,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/rectangle.msh",
}

Output = {
	"Prefix" : "noh_problem",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

SourceTerms = {
	# "source1": {
		# "Function" : "GravitySource",
		# "gravity": 9.8,
		# "source_treatment" : "Explicit",
	# },
	# "source3": {
	# 		"Function": "ExsolutionSource",
	# 		"source_treatment" : "Implicit",
	# },
	"source4": {
		"Function" : "CylindricalGeometricSource",
		# "source_treatment" : "Explicit",
	}
}

InitialCondition = {
	"Function" : "NohProblem",
	"eps": 1e-7,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "SlipWall",
	},
	"x2" : {
		"BCType" : "NohInlet",
		"eps": 1e-7,
	},
	"y1" : {
		"BCType" : "SlipWall",
	},
	"y2" : {
		"BCType" : "SlipWall",
	},
}

# LinkedSolvers = []
# LinkedSolvers = [
# 	{
# 		"DeckName": "conduit.py",
# 		"BoundaryName": "vent",
# 	},
# 	{
# 		"DeckName": "r1r2_cyl.py",
# 		"BoundaryName": "r1",
# 	},
# ]