import numpy as np

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : "PositivityPreserving", #["WENO", "PositivityPreserving"],
	"Solver" : "DG",
}

Mesh = {
	"File" : "../meshes/generalB4.msh",
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
	"ground4" : {
		"BCType" : "SlipWall",
	},
	"symmetry4" : {
		"BCType" : "SlipWall",
	},
	"r4" : {
		"BCType" : "Euler2D2D",
		"bkey": "r4",
	},
	"r3" : {
		"BCType" : "Euler2D2D",
		"bkey": "r3",
	},
}

Output = {
	"Prefix" : "debug_standard3_generalB4",
	"WriteInterval" : 1000,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

LinkedSolvers = [
	{
		"DeckName": "r4r5.py",
		"BoundaryName": "r4",
	},
]