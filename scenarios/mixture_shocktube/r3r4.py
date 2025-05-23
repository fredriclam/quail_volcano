import numpy as np

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : "PositivityPreserving", #["WENO", "PositivityPreserving"],
	"Solver" : "DG",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 1e5,#5e3,
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
}

Mesh = {
	"File" : "../meshes/volcanoA4.msh",
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
	"Prefix" : "referenceC4",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

LinkedSolvers = [
	{
		"DeckName": "r4r5.py",
		"BoundaryName": "r4",
	},
]