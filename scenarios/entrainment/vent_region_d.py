''' Dependent vent region domain for replacing p boundary condition '''

import numpy as np


Numerics = {
	"SolutionOrder" : 0, #0, #1
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	# "ApplyLimiters" : "PositivityPreservingMultiphasevpT", # p0: turned off
	"ArtificialViscosity" : False, #True, # p0: turned off
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 0.3, #0.3, #1 #.3,#0.3,#500,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	# For free surface in P0, L2 projection is needed
	'L2InitialCondition': True,
}

param_suffix = "25_150_15"

Mesh = {
	# "File" : "../meshes/submarine_atmosphere20_large.msh",
	"File" : f"../meshes/sub_atm_{param_suffix}.msh",
}

Output = {
	# "Prefix" : f"coreblowout5A_atm_{param_suffix}",
	# "Prefix" : f"coreblowout7A_atm_{param_suffix}",
	# "Prefix" : f"coreblowout7R_atm_{param_suffix}",
	# "Prefix" : f"coreblowout7D_atm_{param_suffix}",
	# "Prefix" : f"coreblowout7R2_atm_{param_suffix}",
	# "Prefix" : f"coreblowout7D2_atm_{param_suffix}",
	# "Prefix" : f"coreblowout5A_atm_{param_suffix}",
	# "Prefix" : f"coreblowout8E_atm_{param_suffix}",
	# "Prefix" : f"coreblowout8F_atm_{param_suffix}",
	"Prefix" : f"corelayer1C_atm_{param_suffix}",
	"WriteInterval" : 100, # 100
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphaseWLMA",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "num_parallel_workers": 6,
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

# Tracer frac calculation
# 3 kPa saturation (25degC) @ 80%RH: 2.4 kPa water partial pressure
# Partial density: pw / (Rw T) -- 2.4e3 Pa / (461J/(kg K) * 300 K) ~ 0.017kg/m^3
# As mass fraction 0.017 / (1.2 + 0.017) = 0.014 as water vapor
tracer_frac_w = 1e-6 # As mass fraction of liquid, not vapor
tracer_frac_m = 1e-6

InitialCondition = {
  "Function" : "OverOceanIsothermalAtmosphere",
  # "h0": -150,
  "hmin": -100, # "hmax": 1000,
  "hmax": 2501, # "hmax": 1000,
  "psurf": 1.01325e5, # 10 m = 1 bar; # Adding in 50 m of water
  "tracer_frac_w": tracer_frac_w,
  "tracer_frac_m": tracer_frac_m,
}


BoundaryConditions = {
  "r1" : {
    "BCType" : "SlipWall",
  },
  "surface" : {
    "BCType" : "SlipWall",
  },
  "x2": {
    "BCType": "MultiphasevpT2D1D",
    "bkey": "comm2D1D",
  },
  "jetslope" : {
    "BCType" : "SlipWall",
  },
  "symmetry" : {
    "BCType" : "SlipWall",
  },
}

# LinkedSolvers = []
# LinkedSolvers = [
# 	{
# 		"DeckName": "conduit.py",
# 		"BoundaryName": "x2",
# 	},
# 	# {
# 	# 	"DeckName": "r1r2.py",
# 	# 	"BoundaryName": "r1",
# 	# },
# ]