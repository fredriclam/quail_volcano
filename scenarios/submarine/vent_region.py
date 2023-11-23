import numpy as np

# 11: detailed, dt = 1/1400, AV = 1.0 in vent, 5.0 in r1r2

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 10,
	"NumTimeSteps" : 10*700, # 2500 per second is ok # min mesh edge length: 6.2; about 600 is min
	"TimeStepper" : "RK3SR",
}

Numerics = {
	"SolutionOrder" : 1,
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 0.3, #1 #.3,#0.3,#500,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	# For free surface in P0, L2 projection is needed
	# 'L2InitialCondition': True, # Use interpolation instead of L2 projection of Riemann data
	'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Mesh = {
	"File" : "../meshes/submarine_one_block.msh",
}

Output = {
	"Prefix" : "deep_submarine_injection2_atm1",
	# "Prefix" : "submarine_proto_WLMA12_atm1",
	"WriteInterval" : 5,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphaseWLMA",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "num_parallel_workers": 7,
}

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
	},
	# "source3": {
	# 		"Function": "ExsolutionSource",
	# 		"source_treatment" : "Implicit",
	# },
	# "source4": {
	# 	"Function" : "CylindricalGeometricSource",
	# 	# "source_treatment" : "Explicit",
	# }
	# 'source5': {'Function': 'WaterMassSource',
	# 'mass_rate': 0.0,
	# 'specific_energy': 109388.56885035457,
	# 'injection_depth': -50,
	# 'gaussian_width': 50,
	# 'conduit_radius': 50,
	# }
	'source6': {'Function': 'MagmaMassSource',
	'mass_rate': 100.0,
	'injection_depth': -50,
	'gaussian_width': 50,
	'conduit_radius': 50,
	}
}

'''# Log
injection1: watermasssource, 0.0 mass rate
injection2: 'mass_rate': 100.0,
						'injection_depth': -50,
						'gaussian_width': 50,
						'conduit_radius': 50,
injection3:


'''

# Restart = {
# 	"File" : "___.pkl",
# 	"StartFromFileTime" : True
# }

# mode = "WaterLayer"
mode = "WaterLayer"

if "WaterLayer" == mode:
	InitialCondition = {
		"Function" : "IsothermalAtmosphere",
		"h0": -150,
		"hmax": 3000,
	}

	BoundaryConditions = {
		"r1" : {
			"BCType" : "SlipWall",
			# "BCType" : "MultiphasevpT2D2D",
			# "bkey": "r1",
		},
		"ground" : {
			"BCType" : "SlipWall",
		},
		"ground2" : {
			"BCType" : "SlipWall", # Extra section in ground for mesh dx control
		},
		"flare" : {
			"BCType" : "SlipWall",
		},
		"pipewall" : {
			"BCType" : "SlipWall",
		},
		"x2" : {
			"BCType" : "SlipWall",
			# "BCType" : "LinearizedImpedance2D",
		},
		"symmetry" : {
			"BCType" : "SlipWall",
		},
		"symmetry2" : {
			"BCType" : "SlipWall", # Extra section in ground for mesh dx control
		},
	}
elif "DebrisFlow" == mode:
	InitialCondition = {
		"Function" : "DebrisFlow",
	}

	BoundaryConditions = {
		"r1" : {
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
			"BCType" : "SlipWall",
		},
		"symmetry" : {
			"BCType" : "SlipWall",
		},
	}
else:
	raise Exception("Oops mode")

	ExactSolution = InitialCondition.copy()


# LinkedSolvers = []
LinkedSolvers = [
	# {
	# 	"DeckName": "conduit.py",
	# 	"BoundaryName": "vent",
	# },
	# {
	# 	"DeckName": "r1r2.py",
	# 	"BoundaryName": "r1",
	# },
]