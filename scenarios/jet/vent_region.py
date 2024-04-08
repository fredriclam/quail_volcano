import numpy as np
import run_globals

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 60*1,
	"NumTimeSteps" : 60*8000, # P1: 60*1000, # 60*6000
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
	"WriteInterval" : run_globals.write_interval // 2,
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
		"BCType" : "ChokedInlet2D",
    "p_in": 1e5,
    "T_in": 1000,
    "yWv": 0.03,
    "yA": 1e-4,
    "yWt": 0.04,
		"yF": 1-1e-7,
		"yC": 1-1e-7,
	},
	"symmetry" : run_globals.SlipWallQ,
}

BoundaryConditions["r1"] = run_globals.SlipWallQ
# LinkedSolvers = [
# 	{
# 		"DeckName": "r1r2.py",
# 		"BoundaryName": "r1",
# 	},
# ]