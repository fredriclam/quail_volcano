import numpy as np
import run_globals

Numerics = {
	"SolutionOrder" : run_globals.ElementOrder,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
	"AVParameter" : 30,
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : f"../meshes/{run_globals.mesh_prefix}_2.msh",
}

Output = {
	"Prefix" : f"{run_globals.file_prefix}_atm2",
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

extend_atm = True

BoundaryConditions = {
	"ground2" : run_globals.SlipWallQ,
	"symmetry2" : run_globals.SlipWallQ,
	"r2" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r2",
	},
	"r1" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r1",
	},
}
LinkedSolvers = [
	{
		"DeckName": "r2r3.py",
		"BoundaryName": "r2",
	},
]

if not extend_atm:
	BoundaryConditions["r2"] = {
		"BCType" : "LinearizedImpedance2D",
	}
	LinkedSolvers = []