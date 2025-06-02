import numpy as np
import run_globals

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 60*1,
	"NumTimeSteps" : 60*6000, # 60*6000, # 1800 for 1D @ 10 m
	"TimeStepper" : "RK3SR",
}

Numerics = {
	"SolutionOrder" : run_globals.ElementOrder,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
	"AVParameter" : 30,
	'L2InitialCondition': False,
}

Mesh = {
	"File" : f"../meshes/{run_globals.mesh_prefix}_1.msh",
}

Output = {
	"Prefix" : f"{run_globals.file_prefix}_atm1",
	"WriteInterval" : run_globals.write_interval,
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

InitialCondition = run_globals.InitialCondition

ExactSolution = InitialCondition.copy()

BoundaryConditions = {
	"r1" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r1",
	},
	"ground" : run_globals.SlipWallQ,
	"flare" : run_globals.SlipWallQ,
	"pipewall" : run_globals.SlipWallQ,
  "x2" : {
		"BCType" : "MultiphasevpT2D1D",
    "bkey": "x2",
	},
	"symmetry" : run_globals.SlipWallQ,
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
