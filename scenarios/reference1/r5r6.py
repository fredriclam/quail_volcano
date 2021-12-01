import numpy as np

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : "PositivityPreserving", #["WENO", "PositivityPreserving"],
	"Solver" : "DG",
}

Mesh = {
	"File" : "../meshes/generalB6.msh",
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
	"ground6" : {
		"BCType" : "SlipWall",
	},
	"symmetry6" : {
		"BCType" : "SlipWall",
	},
	"r6" : {
		"BCType" : "SlipWall",
	},
	"r5" : {
		"BCType" : "Euler2D2D",
		"bkey": "r5",
	},
}

Output = {
	"Prefix" : "debug_standard3_generalB6",
	"WriteInterval" : 1000,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

LinkedSolvers = []