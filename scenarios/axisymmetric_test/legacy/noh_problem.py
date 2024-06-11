import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 1.0,
	"NumTimeSteps" : 5000*4,
	"TimeStepper" : "RK3SR", # "FE",
}

Numerics = {
	"SolutionOrder" : 1,# 2,
	# "SolutionBasis" : "LagrangeSeg",
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.0,
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True, #True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 30,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

is_1D = True

if is_1D:
	Numerics["SolutionBasis"] = "LagrangeSeg"

if is_1D:
	Mesh = {'ElementShape': 'Segment',
	'File': None,
	'NumElemsX': 2000,
	'xmax': 1,   # Top of the conduit is placed at -150 m for 2D coupling, but you can choose whatever if not coupling to 2D.
	'xmin': 0
	}
else:
	Mesh = {
		"File" : "../meshes/rectangle.msh",
	}

Output = {
	"Prefix" : "noh_problem1D_e7N2000p1c1e3",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

SourceTerms = {
	"source4": {
		"Function" : "CylindricalGeometricSource",
		# "source_treatment" : "Explicit",
	}
}

InitialCondition = {
	"Function" : "NohProblem",
	"eps": 1e-5, #1e-5, , #1e-5, # 1e-7 works fine, N=2000 for unit interval
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"x1" : {
		"BCType" : "SlipWall",
		"use_stagnation_correction": True,
		"Q": 1e2, # Max density factor limiter
	},
	"x2" : {
		"BCType" : "NohInlet",
		"eps": InitialCondition["eps"]
	},
}

if not is_1D:
  BoundaryConditions["y1"] = {
    "BCType" : "SlipWall",
	}
  BoundaryConditions["y2"] = {
    "BCType" : "SlipWall",
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