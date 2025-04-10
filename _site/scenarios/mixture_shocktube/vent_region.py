import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 500*1e-2, # 2*490e-4*25,#0.030,#1.0, #0.1 @ meter scale
	"NumTimeSteps" : 250*100, # 490*25*2,#60,#2000,#1*1000, # 20000,#2*20000*4,#5000*2, #13000*2, #5000 @ meter scale
     # 100000 for generalB1, 400~K
	"TimeStepper" : "FE",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	# "ApplyLimiters" : ["PositivityPreservingMultiphasevpT", "WENO", "PositivityPreservingMultiphasevpT"], "ShockIndicator": "MinMod", "TVBParameter": 0.0,
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 150,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/volcanoA1.msh",
}

Output = {
	"Prefix" : "mixture_shocktube_atm1_cyl",
	"WriteInterval" : 200,
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
		# "source_treatment" : "Explicit",
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

if False:
	# Sod state
	rhoAmbient = 0.125
	pAmbient = 0.1
	eAmbient = pAmbient / (Physics["SpecificHeatRatio"] - 1.0)

InitialCondition = {
	"Function" : "LinearAtmosphere",
	# "state" : UQuiescent,
}

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"r1" : {
		# "BCType" : "SlipWall",
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
		# "BCType" : "SlipWall",
		"BCType" : "MultiphasevpT2D1D",
		"bkey": "vent",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

# LinkedSolvers = []
LinkedSolvers = [
	{
		"DeckName": "conduit.py",
		"BoundaryName": "vent",
	},
	{
		"DeckName": "r1r2_cyl.py",
		"BoundaryName": "r1",
	},
]