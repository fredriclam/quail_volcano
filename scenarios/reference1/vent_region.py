import numpy as np

TimeStepping = {
	"InitialTime" : 0.,
	"FinalTime" : 0.01*200, #0.1 @ meter scale
	"NumTimeSteps" : 1000*40,#5000 @ meter scale
  # 100000 for generalB1, 400~K
	"TimeStepper" : "FE",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["WENO", "PositivityPreserving"],
	"Solver" : "DG",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 1e4,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
}

Mesh = {
	"File" : "../meshes/volcanoC1.msh",
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
	"r1" : {
		"BCType" : "Euler2D2D",
		"bkey": "r1",
	},
	# "r1" : {
	# 	"BCType" : "SlipWall",
	# },
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
		"BCType" : "Euler2D1D",
		"bkey": "x2",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

Output = {
	"Prefix" : "referenceD1",
	"WriteInterval" : 200,
	"WriteInitialSolution" : True,
	"AutoPostProcess": True,
}

LinkedSolvers = [
	{
		"DeckName": "conduit.py",
		"BoundaryName": "x2",
	},
	{
		"DeckName": "r1r2.py",
		"BoundaryName": "r1",
	},
]