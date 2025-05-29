import numpy as np

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 6*60, # 3 for msh1, msh2 for ringfault; 30 for msh3
	"NumTimeSteps" : 6*20000, #FE:48000, # 12000 for msh1 msh2 msh3 # 20 m mesh? 20/1500 = 0.01333 s =1/(75) Hz
	"TimeStepper" : "FE", #"FE", #"RK3SR",
}

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

higher_order = False
if higher_order:
  Numerics["SolutionOrder"] = 1
  Numerics["ArtificialViscosity"] = True
  Numerics["AVParameter"] = 10
  Numerics["ApplyLimiters"] = "PositivityPreservingMultiphasevpT"
  TimeStepping["NumTimeSteps"] *= 2

Mesh = {
	"File" : "../meshes/submarine_conduitjet.msh",
}

Output = {
	# "Prefix" : "ringfault_above_ocean_vent_region4",
	# Restart file "Prefix" : "ringfault_above_ocean_vent_region4_1175plus",
	# "Prefix" : "deep_submarine_crater10blast_atm1", # AGU checkpoint (crater blast)
	# "Prefix" : "submarine_proto_WLMA12_atm1",
	"Prefix" : "entr_calibration2_atm1",
	"WriteInterval" : 20, # 5
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Physics = {
    "Type" : "MultiphaseWLMA",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "num_parallel_workers": 10,
}

surface_height = 300

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
		# "cutoff_height": surface_height, # defaults np.inf
	},
	# "source3": {
	# 		"Function": "ExsolutionSource",
	# 		"source_treatment" : "Implicit",
	# },
	"source4": {
		"Function" : "CylindricalGeometricSource", # Why was this off for "submarine"?
		# "source_treatment" : "Explicit",
	}
	# 'source5': {'Function': 'WaterMassSource',
	# 'mass_rate': 0.0,
	# 'specific_energy': 109388.56885035457,
	# 'injection_depth': -50,
	# 'gaussian_width': 50,
	# 'conduit_radius': 50,
	# }
	# 'source6': {'Function': 'MagmaMassSource',
	# 'mass_rate': 200e6, # next, try: 100e3*2.6e3,
	# 'injection_depth': -350,
	# 'gaussian_width': 50,
	# 'conduit_radius': 50,
	# 'cutoff_height': -150,
	# }
}

# Restart = {
# 	"File" : "ringfault_above_ocean_vent_region4_1175.pkl",
# 	"StartFromFileTime" : True
# }

# Tracer frac calculation
# 3 kPa saturation (25degC) @ 80%RH: 2.4 kPa water partial pressure
# Partial density: pw / (Rw T) -- 2.4e3 Pa / (461J/(kg K) * 300 K) ~ 0.017kg/m^3
# As mass fraction 0.017 / (1.2 + 0.017) = 0.014 as water vapor
tracer_frac_w = 1e-6 # As mass fraction of liquid, not vapor
tracer_frac_m = 1e-6

InitialCondition = {
  "Function" : "OverOceanIsothermalAtmosphere",
  # "h0": -150,
  "hmin": 0, # "hmax": 1000,
  "hmax": 25000, # "hmax": 1000,
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
    "bkey": "x2",
  },
  "symmetry" : {
    "BCType" : "SlipWall",
  },
}

# LinkedSolvers = []
LinkedSolvers = [
	{
		"DeckName": "conduit.py",
		"BoundaryName": "x2",
	},
	# {
	# 	"DeckName": "r1r2.py",
	# 	"BoundaryName": "r1",
	# },
]