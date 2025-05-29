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
	"File" : f"../meshes/{run_globals.mesh_prefix}_7.msh",
}

Output = {
	"Prefix" : f"{run_globals.file_prefix}_atm7",
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
i = 7

BoundaryConditions = {
	f"ground{i}" : run_globals.SlipWallQ,
	f"symmetry{i}" : run_globals.SlipWallQ,
	f"r{i}" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": f"r{i}",
	},
	f"r{i-1}" : {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": f"r{i-1}",
	},
}
LinkedSolvers = [
	{
		"DeckName": f"r{i}r{i+1}.py",
		"BoundaryName": f"r{i}",
	},
]

if not extend_atm:
	BoundaryConditions[f"r{i}"] = {
		"BCType" : "LinearizedImpedance2D",
	}
	LinkedSolvers = []