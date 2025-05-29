import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 9.0,
	"NumTimeSteps" : 573 , # 3*9*5*2,
	"TimeStepper" : "RK3SR",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	# "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
	"AVParameter" : 30,
	'L2InitialCondition': False,
}

Mesh = {
	"File" : f"../meshes/sphere_uniform_h120.msh",
}

Output = {
	"Prefix" : "sphere_h120p1", # mesh sphere_uniform2 (finer, smaller) @ 5 Hz: "sphere_5" # 5 Hz: "sphere_4"    # 10 Hz: "sphere_3",
	"WriteInterval" : 191,
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