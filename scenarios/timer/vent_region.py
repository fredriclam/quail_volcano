import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.1, #*0.1000, #0.0014, #0.0014625, # 0.0250,
	"NumTimeSteps" : 10000, #8000 ,#560, #585, #400,
  # 100000 for generalB1, 400~K
	"TimeStepper" : "FE",
}

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["WENO", "PositivityPreserving"],
	"Solver" : "DG",
}

Mesh = {
	"File" : "../meshes/generalB1.msh",
}

Physics = {
	"Type" : "Euler",
	"ConvFluxNumerical" : "Roe",
	"GasConstant" : 287,
	"SpecificHeatRatio" : 1.4,
}

rhoAmbient = 1.2
TAmbient = 300.0
eAmbient = rhoAmbient * Physics["GasConstant"] / (Physics["SpecificHeatRatio"] - 1.0) * TAmbient
pAmbient = rhoAmbient * Physics["GasConstant"] * TAmbient

if False:
	# Sod state
	rhoAmbient = 0.125
	pAmbient = 0.1
	eAmbient = pAmbient / (Physics["SpecificHeatRatio"] - 1.0)

UQuiescent = np.array([rhoAmbient, 0.0, 0.0, eAmbient])
InitialCondition = {
	"Function" : "Uniform",
	"state" : UQuiescent,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	# "r1" : {
	# 	"BCType" : "Euler2D2D",
	# 	"bkey": "r1",
	# },
	"r1" : {
		"BCType" : "SlipWall",
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
	# "x2" : {
	# 	"BCType" : "Euler2D1D",
	# 	"bkey": "x2",
	# },
	"x2" : {
		"BCType" : "SlipWall",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

Output = {
	"Prefix" : "debug_standard3",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": True,
}

LinkedSolvers = [
	# {
	# 	"DeckName": "conduit.py",
	# 	"BoundaryName": "x2",
	# },
	# {
	# 	"DeckName": "r1r2.py",
	# 	"BoundaryName": "r1",
	# },
]