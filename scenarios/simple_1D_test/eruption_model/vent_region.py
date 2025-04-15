# 2D axisymmetric part near the vent

# TimeStepping is set in input_file.py

import run_globals

Numerics = {
	"SolutionOrder" : run_globals.ElementOrder2D,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
	"AVParameter" : 30,
	'L2InitialCondition': False, # False == Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : f"../../meshes/{run_globals.mesh_prefix}1.msh",
}

Output = {
	"Prefix" : f"{run_globals.output_file_prefix}_atm1",
	"WriteInterval" : run_globals.write_interval_2D,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

# Source terms: exsolution, fragmentation are not turned on
SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
		'source_treatment': 'Explicit',
	},
	"source2": {
		"Function" : "CylindricalGeometricSource",
		'source_treatment': 'Explicit',
	}
}

# Set initial condition specified in run_globals.py
InitialCondition = run_globals.InitialCondition

# Dummy function passed to quail
ExactSolution = InitialCondition.copy()

# Set boundary conditions (and provisionally add impedance BC at outer boundary)
BoundaryConditions = {
	"r1" : {
		"BCType" : "LinearizedImpedance2D",
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
		"BCType" : "MultiphasevpT2D1D",
		"bkey": "vent",
	},
	"symmetry" : {
		"BCType" : "SlipWall",
	},
}

# Setting to extend atmosphere here
# When True, attaches simulation r1r2.py to make a bigger 2D domain
# When False, leaves the impedance boundary condition at r1 as is
extend_atm = False

if extend_atm:
	BoundaryConditions["r1"] = {
		"BCType" : "MultiphasevpT2D2D",
		"bkey": "r1",
	},
	LinkedSolvers = [
		{
			"DeckName": "r1r2.py",
			"BoundaryName": "r1",
		},
	]
else:
	LinkedSolvers = []