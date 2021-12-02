import numpy as np

Numerics = {
	"SolutionOrder" : 0,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : "PositivityPreserving", #["WENO", "PositivityPreserving"],
	"Solver" : "DG",
}

Mesh = {
	"File" : "../meshes/volcanoA5.msh",
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
	"ground5" : {
		"BCType" : "SlipWall",
	},
	"symmetry5" : {
		"BCType" : "SlipWall",
	},
	"r5" : {
		"BCType" : "Euler2D2D",
		"bkey": "r5",
	},
	"r4" : {
		"BCType" : "Euler2D2D",
		"bkey": "r4",
	},
}

Output = {
	"Prefix" : "referenceA5",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

LinkedSolvers = [
	{
		"DeckName": "r5r6.py",
		"BoundaryName": "r5",
	},
]