''' Sample input file for conduit in Vulcanian eruption. See below for instructions.

Sample run:
  cd C:\Volcano\quail_volcano\scenarios\entrainment
  python ../../src/quail core_leak.py
'''



''' Changelog

Old changelog at bottom.

coreleak1A_25_50_15: initial run
coreleak1C_25_50_15: no entrainment and fixed 2D part output; there is no B
coreleak1D_25_50_15: commented out injection; water_depth back to 100; porosity from 1 to 0.5; entrainment_coeff still 0.0
coreleak1E_25_50_15: porosity 0.5 to 0.7 ;T_porous 600 to 473.15, properly 25-50-15 mesh, hwm -150 to -50;
coreleak1F_25_50_15: T_porous 473.15 to 1073.15
coreleak1G_25_50_15: T_porous 1073.15 to 873.15, 10 to 30 seconds; WriteInterval 200 to 100 to match 2D axisymm.
H:  is in core_leak_fork
coreleak1I_25_50_15: T_porous 873.15 to 773.15, hwm -50 to -150 (matched); porosity 0.7 to 0.85 
coreleak1J_25_50_15: *** SUS ***    T_porous 773.15, hwm -50 to -150 (matched); porosity 0.85 to 0.9 ; WriteInterval 100 to 50 IN BOTH! 30 s -> 60 s
coreleak1K_25_50_15: T_porous 773.15, hwm -150 to -50; porosity remains 0.9 ; Still WriteInterval 50, 60 s
coreleak1L_25_50_15: *** IN FORK *** T_porous 773.15, hwm -150 to 0; porosity remains 0.9 ; Still WriteInterval 50, 60 s
coreleak1M_25_50_15: T_porous 773.15 to 573.15; hwm remains -50; porosity 0.9 to 0.99
coreleak1N_25_50_15: *** PLANNED ***  T_porous 773.15 to 673.15; hwm remains -50; porosity 0.99 as in M
coreleak1O_25_50_15: *** PLANNED ***  T_porous 773.15 to 473.15; hwm remains -50; porosity 0.99 as in M
coreleak1P_25_50_15: *** PLANNED ***  as J, but with 150 mesh


Want to change:
2 km 800 elems to 1 km 400 elems, 1000 steps per second to 500

Ideas:
plug at depth representing the fracture of a hydrothermal seal?
'''

import numpy as np

output_prefix = "coreleak1N_25_50_15"

# Key parameters
hsurf = 0.0               # Height of top boundary
hwm = -50                # Height of vent *************** (NEGATIVE) e.g. -150 --- below the top of domain, which is z = -water_depth
conduit_radius = 25       # 1D cross-sectional area ***********
mu0 = 1e5                 # Viscosity (Pa s) if constant viscosity model
coeff_entrainment = 0.0   # Entrainment coefficient
water_depth = 100          # Depth of overlying water layer
p_boundary = 100000.0 + water_depth*1000*9.8 # Pressure at top boundary
jet_angle_deg = 15        # Underwater jet angle in degrees *****************
z_premix = hwm - 250
z_premix = -2000          # Remove magma layer ****************************

# Initial condition
T_magma = 1100 + 273.15
T_water = 300
T_porous = 400+273.15 # 1273.15 # 600 #  + 273.15
porosity = 0.99 # 0.5 # 0.3 # in premix region
plug_delta_p = 2.5e6 # 2.5e6 # 10e6
# Chamber boundary condition if not wall
p_chamber = 55918359.19099045 # (10e6) # 76094058.10135078 (30e6) # 55918359.19099045 (10e6) # 46880035.763866425 (10e6) # 47224934.806827545 # 46880035.763866425 # 47224934.806827545 # 55e6
T_chamber = T_magma
yC = 0.0          # Crystal mass fraction of chamber and initial
yWt = 0.02 # 0.03        # Total water mass fraction (constrained by collapse-favored scenario?)
chi_water = (1.0 - yC) * yWt / (1 - yWt) # Water concentration

Output = {'AutoPostProcess': False,
  "Prefix" : output_prefix,
  'WriteInitialSolution': True,
  'WriteInterval': 50, # 25,
}

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 60, # 20,
	"NumTimeSteps" : 60*1000, # 20*1000,
	"TimeStepper" : "FE",
}

Numerics = {
  'AVParameter': 0.3,
  # 'ApplyLimiters': 'PositivityPreservingMultiphasevpT',
  'ArtificialViscosity': False, #True,
  'L2InitialCondition': True,
  'SolutionBasis': 'LagrangeSeg',
  'SolutionOrder': 0, #1,
  'Solver': 'DG'}

Mesh = {
  'ElementShape': 'Segment',
  'File': None,
  'NumElemsX': 800, # 4000,
  'xmax': -0.0,
  'xmin': -2000.0
}

Physics = {
    "Type" : "MultiphaseWLMA",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "num_parallel_workers": 8,
    "Liquid": {
      "K": 10e9,
      "rho0": 2.6e3,
      "p0": 3.6e6,
      "E_m0": 0,
      "c_m": 1e3,
    }
}

SourceTerms = {
  'source1': {
    'Function': 'GravitySource',
    'gravity': 9.8,
    'source_treatment': 'Explicit'
  },
  # 'source2': {
  #   'Function': 'FrictionVolFracConstMuLayer',   # Friction source term for given conduit radius
  #   'conduit_radius': conduit_radius,
  #   'mu': mu0,
  #   'water_depth': np.abs(hwm),
  #   'source_treatment': 'Explicit',
  # },
  #  'source4': {
  #     'Function': 'FragmentationStrainRateSource',
  #     'fragsmooth_scale': 0.05,        # Fragmentation smoothing scale (for two-sided smoothing)
  #     'source_treatment': 'Explicit',
  #     'tau_f': 0.01,
  #     'G': 1e9,
  #     'k': 0.1,
  #     "which_criterion": "both",
  #     "conduit_radius": conduit_radius,
  #     "shear_factor": 1.0,
  #   },
  'source5': {
    'Function': 'WaterEntrainmentSource',
    "entrainment_coefficient": coeff_entrainment,
    # "injection_depth": hwm,
    "conduit_radius": conduit_radius,
		"h1": hwm,
    "T_ref": 300,
    "p_ref": 1e5,
    'theta_deg': jet_angle_deg,           # degrees
  }, # TODO: add variable geometry through hydraulic diameter
  
  # 'source7': {
  #   'Function': 'WaterMassSource',
  #   'source_treatment': 'Explicit',
  #   'mass_rate': 1e5,
  #   'specific_energy': 109388.56885035457,
  #   'injection_depth': -350,
  #   'gaussian_width': 50,
  #   'conduit_radius': 25,
  #   'hwm': -50,
  #   'injection_time_start': 0.0,
  #   'injection_time_end': 2.0,
  # },

  'source6': {
    'Function': 'AreaGeometricSource',
    'h1': hwm,
    'r1': conduit_radius,
    'theta1_deg': jet_angle_deg,           # degrees
    'added_depth': water_depth, # >= 0
    'mode': "jet",
  }
}

# Restart = {
# 	"File" : "file_name.pkl",
# 	"StartFromFileTime" : True
# }

# Define a traction function
def gaussian_traction(x:np.array, total_pressure_change=10e6, x0=-350, sigma=50) -> np.array:
  ''' Traction function added to the hydrostatic equation. Negative
   sign indicates downward traction on the fluid. Units are Pa / m. 
   Total pressure change due to traction is amp * sigma.
   Inputs:
     x: array of points at which traction is evaluated
     total_pressure_change: total pressure change across the traction. The
       Gaussian amplitude is calculated from this.
     x0: Gaussian center (m)
     sigma: standard deviation parameter (length scale of traction function)
   '''
  # Compute amplitude of Gaussian TODO:
  amplitude = total_pressure_change / (np.sqrt(np.pi) * sigma)
  _t = (x-x0)/sigma
  return -amplitude * np.exp(-_t * _t)

# Initial condition parameters
# Note that some parameters here are repeated, and must be consistent with the
# parameters specified in SourceTerms. It's hardcoded here because this
# file was generated by a script that writes parallelized input files.
# InitialCondition = {
#   'Function': 'UnderwaterMagmaConduit',
#   "T_magma": T_magma,
#   "T_water": T_water,
#   "psurf": 1.01325e5, # Pressure at ocean surface
#   "hsurf": hsurf,
# 	"hwm": hwm, #-50, #Water magma interface height
#   "hmin": Mesh["xmin"],#-2050,
#   "delta_p": plug_delta_p,
#   "yC": yC,
#   "yWt": yWt,
# 	"solubility_k": 8e-6, #!
#   "solubility_n": 0.5,
# }

InitialCondition = {
  'Function': 'ThreeLayerModel',
  "T_magma": T_magma,
  "T_water": T_water,
  "T_infiltrate": T_porous,
  "delta_p": plug_delta_p,
  "psurf": 1.01325e5, # Pressure at ocean surface
  "z_ocean_surface": water_depth,
	"z_vent": hwm, #-50, #Water magma interface height
  "z_premix": z_premix,
  "porosity": porosity,
  "z_min": Mesh["xmin"], #-2050,
  "yC": yC,
  "yWt": yWt,
  "tracer_frac": 1e-4,
  "solubility_k": 5e-6, # 8e-6,
  "solubility_n": 0.5,
}

# This is needed by Quail. This is not the exact solution, just something callable.
ExactSolution = {'Function': 'UnderwaterMagmaConduit'}

# Set boundary conditions here.
BoundaryConditions = {
#   'x1': {'BCType': 'PressureStableLinearizedInlet1D',   # Inlet boundary condition
#         'T_chamber': T_chamber,                         # Chamber temperature
#         'cVFamp': 0.0,                                  # Amplitude of crystal fraction variation
#         'cVFav': yC,                                    # Crystal fraction input
#         'chi_water': chi_water,                         # Mass concentration of water
#         'cos_freq': 0.25,                               # Frequency of crystal fraction variation
#         'is_gaussian': False,                           # If true, use Gaussian variation instead of cosine
#         'p_chamber': p_chamber,                         # Chamber pressure
#         'trace_arho': 2.6e-04,           # Trace density (for numerical stability)
#         # 'solubility_k': 5e-06,                          # Henry's law coefficient
#         # 'solubility_n': 0.5,                            # Henry's law exponent
# },
  'x1': {
    "BCType": "SlipWall",
  },
  # For running serial, using p boundary condition:
 'x2': {'BCType': 'PressureOutlet1D',
        'p': p_boundary,  # atm + 100 m water
        },
}

# Linked parallel solvers. If running in serial, leave as empty list.
LinkedSolvers = []

use_1D2D_link:bool = True

if use_1D2D_link:
  BoundaryConditions['x2'] = {
    'BCType': 'MultiphasevpT2D1D',                   # Pressure outlet boundary condition (automatically chokes if needed)
    "bkey": "comm2D1D"                                   # Boundary pressure (if flow not choked)
  }
  LinkedSolvers = [{'BoundaryName': 'comm2D1D',
                  'DeckName': 'vent_region_cleak.py'}]

'''
entrainment1
entrainment2: 'NumElemsX': 1000 -> 2000, NumTimeSteps  10*1200 -> 10*2400, 'WriteInterval': 10 -> 20,
entrainment3: 'NumElemsX': 2000 -> 1000, NumTimeSteps  10*2400 -> 10*1200, 'WriteInterval': 20 -> 30,
entrainment4:  fix inlet boundary condition for WLMA EOS, terminated
entrainment4_f:  fix inlet boundary condition for WLMA EOS, 2000 Elems, 2400 NumTimeSteps
entrainment5:  'NumElemsX': 2000 -> 4000, NumTimeSteps  10*2400 -> 10*4800, 'WriteInterval': 30,
entrainment6_a:  'NumElemsX': 2000, NumTimeSteps  10*2400, 'WriteInterval': 30, plug_delta_p = 0 -> 5e6
entrainment6_b:  'NumElemsX': 2000, NumTimeSteps  10*2400, 'WriteInterval': 30, plug_delta_p = 0 -> 10e6
entrainment6_c:  'NumElemsX': 2000, NumTimeSteps  10*2400, 'WriteInterval': 30, plug_delta_p = 0 -> 10e6, conduit_radius = 5 -> 25
New generation:
entrainment7: Updated source. 10 MPa overpressure.
entrainment8: Updated angle 20 deg. 10 MPa overpressure.
entrainment9: Updated angle 20 deg. 10 MPa overpressure. corrected drag shutoff location (sign error)
entrainment10: Coupled to 2D
entrainment10a: Coupled to 2D, longer; last time that hwm = -150, conduit_radius = 25, jet_angle_deg = 15?
entrainment11: Depth up from  20 to 10, writeinterval 300 -> 30, FinalTime 60->10
  See parameter table

coreblowout1: Removed entrainment term. Added WaterMassSource. Also changed heat capacity -> 1e3 from 3e3. Timecale ~2 s for addition. TEST
coreblowout2: Initial condition 3-layer model  (coreblowout2_5_50_15) p_chamber == 47224934.806827545.
coreblowout3: Initial condition 3-layer model, (coreblowout3_25_150_15)depth+ p_chamber == 46880035.763866425
coreblowout4: T:600 -> 800 K, (coreblowout4_25_150_15) p_chamber == 46880035.763866425 (used 47.22... MPa instead...)
coreblowout5: T:800 -> 700 K, (coreblowout5_25_150_15) p_chamber == 46880035.763866425
coreblowout5A: refined 1D NumElemsX 2000 -> 4000, WriteInterval 100 -> 200, FinalTime 20 -> 40,  NumTimeSteps 20*2000 -> 40*4000
               coreblowout5A_25_150_15
coreblowout5B: refined 1D NumElemsX 2000 -> 1000, WriteInterval 100 -> 50, FinalTime 20 -> 10,  NumTimeSteps 20*2000 -> 10*1000
               coreblowout5B_25_150_15
               porosity 0.3 -> 0.5, yWt 0.03 -> 0.02
6B_25_50_15: shallower, with higher gas pressure 10e6 -> 20e6, p_chamber == 55918359.19099045 [INVALID RUN]
7A_25_150_15: back to total 250 m depth, with gas pressure 10e6, SlipWall,
              T_porous = 600 K 
              TimeSteps 60*2000, NumElemsX==2000 (1 per m), WriteInterval 100
7B_25_150_15: faster, check sound speed in water layer @ 120*1000, 1000 elemsX
7C_25_150_15: faster, check sound speed in water layer @ 120*1000, 1000 elemsX, big timestepping
7R_25_150_15: air jet replacement 600 deg C
7D_25_150_15: no friction, exsolution 600 deg C
7R2_25_150_15: R but 600 K, no friction, exsolution -- retroactively changed              Time Interval  20*500 -> 20*1000
               Removed pressure interface term
7D2_25_150_15: R but 600 K, no friction, exsolution -- removed water layer. role swapped? Time Interval  20*500 -> 20*1000
               Removed pressure interface term
coreblowout8E_25_50_15: as 7R2_25_150_15 but with shallower water layer 7R2_25_50_15
coreblowout8F_25_150_15: as 7R2_25_150_15 but with shallower water layer + 2.5 MPa -> 10 MPa; WriteInterval 25 -> 200; NO ENTRAINMENT
coreblowout8G_25_150_15: as 8G but with zero porosity + 2.5 MPa; WriteInterval 200; ENTRAINMENT; T = 1000 + 273.15 K
coreblowout8H_25_150_15: as 8G but with zero porosity + 2.5 MPa; WriteInterval 200; ENTRAINMENT
'''