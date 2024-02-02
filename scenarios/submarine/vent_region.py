import numpy as np

# 11: detailed, dt = 1/1400, AV = 1.0 in vent, 5.0 in r1r2

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 30, # 3 for msh1, msh2 for ringfault; 30 for msh3
	"NumTimeSteps" : 12000, # 12000 for msh1 msh2 msh3
	"TimeStepper" : "FE", #"FE", #"RK3SR",
}

Numerics = {
	"SolutionOrder" : 0, #0, #1
	"SolutionBasis" : "LagrangeTri",
	"Solver" : "DG",
	"ApplyLimiters" : "PositivityPreservingMultiphasevpT",
	"ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 0.3, #0.3, #1 #.3,#0.3,#500,#5e3
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
	# For free surface in P0, L2 projection is needed
	# 'L2InitialCondition': True, # Use interpolation instead of L2 projection of Riemann data
	'L2InitialCondition': True, # False, # Use interpolation instead of L2 projection of Riemann data # TODO: check this. using now True if p==0
}

Mesh = {
	"File" : "../meshes/submarine_ringfault4.msh",
}

Output = {
	"Prefix" : "deep_submarine_ringfault10_atm1",
	# "Prefix" : "deep_submarine_crater10blast_atm1", # AGU checkpoint (crater blast)
	# "Prefix" : "submarine_proto_WLMA12_atm1",
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
		"cutoff_height": surface_height,
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
	# 'source6': {'Function': 'MagmaMassSource',
	# 'mass_rate': 200e6, # next, try: 100e3*2.6e3,
	# 'injection_depth': -350,
	# 'gaussian_width': 50,
	# 'conduit_radius': 50,
	# 'cutoff_height': -150,
	# }
}

'''# Log
injection1: watermasssource, 0.0 mass rate
injection2: 'mass_rate': 100.0,
						'injection_depth': -50,
						'gaussian_width': 50,
						'conduit_radius': 50,
injection4: no uplift, 2D conduit approximation
						'mass_rate': 1000000.0,
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection5_atm1: *complete*
						'mass_rate': 1000000.0,
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection6_atm1:
						'mass_rate': 5000000.0,
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection7_atm1:
						'mass_rate': 50e6, # O(50e3) m^3
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection8_atm1:
						'mass_rate': 25e6, # O(25e3) m^3
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
						3* dt multiplier: crashed @ 902: Time = 4.29524
deep_submarine_injection9_atm1:
						2* dt:
deep_submarine_injection10_atm1:
						back to 3* dt:
						'mass_rate': 10e6,
deep_submarine_injection11_atm1:
						same mass rate as 10, but wider:
						'mass_rate': 10e6,
						'injection_depth': -350,
						'gaussian_width': 100,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection12_atm1: narrow, 50% higher mass
						'mass_rate': 15e6,
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection13_atm1: garbage (as 12, but with lower pressure at top; but density is liquid, leading to ??? at top)
						,
deep_submarine_injection14_atm1: as 12, but with lower pressure cutoff height for buffer region [STOPPED]
						'mass_rate': 15e6,
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection15_atm1: (as 12, but with lower pressure at top) cutoff-heighted [CRASHED during timestep]
						'mass_rate': 200e6,
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection16_atm1: cutoff-heighted, higher mass, p0+FE
						'mass_rate': 200e6,
						'injection_depth': -350,
						'gaussian_width': 50,
						'conduit_radius': 50,
						'cutoff_height': -150,
deep_submarine_injection17_atm1: cutoff-heighted, higher mass, p0+FE (NEXT: TODO: gaussian_Width = 25 as below)
						'mass_rate': 200e6,
						'injection_depth': -350,
						'gaussian_width': 25,
						'conduit_radius': 50,
						'cutoff_height': -150,


The above injection scenarios suffer from ambiguity in interpreting the
pressure wavefield. Injecting mass isochorically increases the pressure globally
after the traversal of the acoustic waves to accommodate the added mass.
The added mass in (17) results in ~3 MPa increase in pressure globally without
signifcant boiling.

Below, we consider premixed hot magma and water in a Riemann problem setup to
avoid both mixing and closed-system injection pressure.

deep_submarine_premix1_atm1: p0+FE (-300 to -150)
	select_yw = 0.95
	p0, Tw0, Tm0 = (9800000.0, 300, 1000)
	(array([1.032568996254891e-04, 9.809405464421462e+02,
        5.162844981274459e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 2.645152678731813e+08,
        9.809406496990458e+02, 1.032568996254891e-04,
        5.162844981274459e+01])

deep_submarine_premix2_atm1: p0+FE (-300 to -150)
	select_yw = 0.93
	(array([1.045819302979261e-04, 9.726119517707126e+02,
        7.320735120854820e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 3.283211682949423e+08,
        9.726120563526429e+02, 1.045819302979261e-04,
        7.320735120854820e+01])

deep_submarine_premix3_atm1: p0+FE (-300 to -150)
	select_yw = 0.94
	(array([1.039151912411363e-04, 9.768027976666815e+02,
        6.234911474468186e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 2.962148278533422e+08,
        9.768029015818727e+02, 1.039151912411363e-04,
        6.234911474468186e+01])

deep_submarine_premix4_atm1: p0+FE (-300 to -150)
	shallower:
	decreased from "hmax": 1000 to "hmax": 350
	select_yw = 0.95
	p0, Tw0, Tm0 = (9800000.0, 300, 1000)
	(array([1.032568996254891e-04, 9.809405464421462e+02,
        5.162844981274459e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 2.645152678731813e+08,
        9.809406496990458e+02, 1.032568996254891e-04,
        5.162844981274459e+01])
	

The following runs put a premix in a crater, and use p0+FE
deep_submarine_crater_atm1:
select_yw = 0.95
	p0, Tw0, Tm0 = (9800000.0, 300, 1000)
	(array([1.032568996254891e-04, 9.809405464421462e+02,
        5.162844981274459e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 2.645152678731813e+08,
        9.809406496990458e+02, 1.032568996254891e-04,
        5.162844981274459e+01])
"hmax": 350,
terminated due to disk space

deep_submarine_crater_atm2:
select_yw = 0.95
	p0, Tw0, Tm0 = (9800000.0, 300, 1000)
	(array([1.032568996254891e-04, 9.809405464421462e+02,
        5.162844981274459e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 2.645152678731813e+08,
        9.809406496990458e+02, 1.032568996254891e-04,
        5.162844981274459e+01])
"hmax": 150,

deep_submarine_crater_atm3:
select_yw = 0.95
	p0, Tw0, Tm0 = (9800000.0, 300, 1000)
	(array([1.032568996254891e-04, 9.809405464421462e+02,
        5.162844981274459e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 2.645152678731813e+08,
        9.809406496990458e+02, 1.032568996254891e-04,
        5.162844981274459e+01])
"hmax": 550,

deep_submarine_crater_atm4:
select_yw = 0.95
	p0, Tw0, Tm0 = (9800000.0, 300, 1000)
	(array([1.032568996254891e-04, 9.809405464421462e+02,
        5.162844981274459e+01, 0.000000000000000e+00,
        0.000000000000000e+00, 2.645152678731813e+08,
        9.809406496990458e+02, 1.032568996254891e-04,
        5.162844981274459e+01])
"hmax": 750,

cleanup:

deep_submarine_crater_atm5: fixed issue where mass was added at x == 0 (using x <= 0 instead of x < 0 condition),
	fixed averaging issue near x = 0(using interpolation instead of projection),
	fixed buffer region pressure too low (clipping yA variable in multiphaseWLMA.py) 
	higher spatial resolution for initial blast (mesh submarine_crater2)
	remains P0 elements

select_yw = 0.90
	p0, Tw0, Tm0 = (9800000.0, 300, 1000)
	array([1.066344900380386e-04, 9.597104103423474e+02,
        1.066344900380386e+02, 0.000000000000000e+00,
        0.000000000000000e+00, 4.271607084743886e+08,
        9.597105169768374e+02, 1.066344900380386e-04,
        1.066344900380386e+02])
	(p_mixed = 72314319.43781662)
did not work, so using
select_yw = 0.95 as above.
"hmax": 1000,

deep_submarine_crater_atm6
select_yw = 0.95 as above.
Now locked both buffer zone height and p = 1e5 height to surface_height (previously buffer zone was at 1000)
"hmax": 500,

deep_submarine_crater_atm7
select_yw = 0.95 as above.
surface_height = 250

deep_submarine_crater_atm8
select_yw = 0.95 as above.
surface_height = 100

_atm8blast
select_yw = 0.50 ***
surface_height = 1000

deep_submarine_crater_atm9blast
select_yw = 0.5
surface_height = 1000
array([1.444294213749651e-04, 7.221471068748257e+02,
        7.221471068748257e+02, 0.000000000000000e+00,
        0.000000000000000e+00, 2.247148483471262e+09,
        7.221472513042470e+02, 1.444294213749651e-04,
        7.221471068748257e+02])

deep_submarine_crater_atm10blast
select_yw = 0.75
surface_height = 1000
U_premix = np.array([1.182373142717186e-04, 8.867798570378898e+02,
        2.955932856792966e+02, 0.000000000000000e+00,
        0.000000000000000e+00, 9.858863765404476e+08,
        8.867799752752039e+02, 1.182373142717186e-04,
        2.955932856792966e+02])


				
New: ring fault type setup

select_yw = 0.75
surface_height = 300
U_premix = np.array([1.182373142717186e-04, 8.867798570378898e+02,
        2.955932856792966e+02, 0.000000000000000e+00,
        0.000000000000000e+00, 9.858863765404476e+08,
        8.867799752752039e+02, 1.182373142717186e-04,
        2.955932856792966e+02])

2:
mesh2 (larger, BCs farther away)

3:
IsothermalAtmosphere changed from
	Uq = np.where((x[:,:,1:2] >= -999999) & (x[:,:,1:2] < 0) , U_premix, Uq)
to
	Uq = np.where((x[:,:,1:2] >= -50) & (x[:,:,1:2] < 0) , U_premix, Uq)
and using mesh2.

"WriteInterval" changed from 2 to 5

4:
IsothermalAtmosphere changed to place magma using
	Uq = np.where((x[:,:,1:2] >= -999999) & (x[:,:,1:2] < -50) , U_chamber, Uq)
and using *mesh1*.
"WriteInterval" changed from 5 to 20
from (1 s)*4000 timesteps to 3 * 4000 timesteps, for finaltime 3 s

5:
mesh2

6:
mesh1, magma-air counterfactual: 10 MPa overpressure

7:
mesh3: scaled up, roof block aspect ratio from Le M/evel et al., refinement change

8:
mesh4: hunga scaling, with magmastatic initial condition

9:
removing premix section (Force was ~100e13 N, compared to 1e13 inferred)

10:
as 9, adding back premix section

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
		# "h0": -150,
		"hmin": -1150,
		"hmax": surface_height, # "hmax": 1000,
		"p_surf": 1e5, # 10 m = 1 bar; 
	}

	BoundaryConditions = {
		"r1" : {
			"BCType" : "SlipWall",
		},
		"groundouter" : {
			"BCType" : "SlipWall",
		},
		"flareouter" : {
			"BCType" : "SlipWall",
		},
		"dikeouter" : {
			"BCType" : "SlipWall",
		},
		"chamberwallouter" : {
			"BCType" : "SlipWall",
		},
		"chambersymmetry" : {
			"BCType" : "SlipWall",
		},
		"chamberwallinner" : {
			"BCType" : "SlipWall",
		},
		"dikeinner" : {
			"BCType" : "SlipWall",
		},
		"flareinner" : {
			"BCType" : "SlipWall",
		},
		"groundinner" : {
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