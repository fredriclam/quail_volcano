import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 3.0,
	"NumTimeSteps" : 3*9*5*2,
	"TimeStepper" : "RK3SR",
}

Numerics = {
	"SolutionOrder" : 2,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	# "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
	"AVParameter" : 30,
	'L2InitialCondition': False,
}

Mesh = {
	"File" : f"../meshes/sphere_uniform_h85.msh",
}

Output = {
	"Prefix" : "sphere_h85", # mesh sphere_uniform2 (finer, smaller) @ 5 Hz: "sphere_5" # 5 Hz: "sphere_4"    # 10 Hz: "sphere_3",
	"WriteInterval" : 90,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
  "CompressedOutput": True,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

SourceTerms = {
	"source1": {
		"Function" : "CylindricalGeometricSource",
	}
}

InitialCondition = {
	"Function" : "UniformAir",
  "T": 300.,
  "p": 1e5,
  "R": 287,
  "c_v": 717.5000000000001,
}

ExactSolution = InitialCondition.copy()

SlipWall = {
  "BCType" : "SlipWall",
  "use_stagnation_correction": True,
  "Q": 10.0,
}

BoundaryConditions = {
	"r1" : {
		"BCType" : "OscillatingSphere",
    "linear_freq": 2.0, # 10.0,
    "u_amplitude": 1.0,
  },
  "r2": {
		"BCType" : "LinearizedImpedance2D",
	},
	"symmetrylower" : SlipWall,
	"symmetryupper" : SlipWall,
}