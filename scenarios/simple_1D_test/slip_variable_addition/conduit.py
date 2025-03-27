
TimeStepping = {
	"InitialTime"  : 0.0,
	"FinalTime"    : 20,      # seconds
	"NumTimeSteps" : 2000,
	"TimeStepper"  : "Strang",
}

Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "ArtificialViscosity" : True,
	"AVParameter" : 90, #0.3,
    'L2InitialCondition': True,
}

Output = {
	"Prefix" : "short_plug_dynamic",
	"WriteInterval" : 40,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 200,
    "xmin" : -1000.0,
    "xmax" : 0.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "Liquid": {"K": 1e9,     # Condensed phase bulk modulus (Pa)
               "rho0": 2.6e3, # Condensed phase density (kg/m^3)
               "p0": 5e6,     # Condensed phase EOS reference pressure (Pa)
               "E_m0": 0,     # Leave zero
               "c_m": 1e3},   # Condensed phase heat capacity (J / (kg K))
}

# Here we specify the state variables corresponding to mass,
# and velocity u and temperature T. From velocity u and the
# mass variables, momentum is calculated internally. The total
# energy is calculated internally from temperature and all
# other provided variables.
InitialCondition = {
    "Function": "LinearPressureGrad",
    "arhoA": 1e-10,    # Mass air per mixture volume
    "arhoWv": 1e-10,   # Mass water in exsolved state
    "u": 1e-10,        # Velocity
    "T": 1000.0,     # Temperature
    "arhoWt": 1e-10, # Mass total water (exsolved + dissolved) per mixture volume
    "arhoCPlug": 50,        # Mass crystals per mixture volume in the plug
    "arhoF": 1e-10,          # Mass fragmented magma per mixture volume
    "rhoSlip": 1e-10,          # Newly implemented state
    "pL": 4.1e6,           # Pressure on the left boundary [M Pa]
    "pL_plug": 4.1e6,      # Pressure on left boundary of the plug
    "pR": 1e5,             # Pressure on the right boundary [M Pa]
    "x_plug": 900,         # Position of the plug from the left of the domain [m]
    "x_length": 1000,      # Length of the plug [m]
}

SourceTerms = {
    #'source2': {
    #    'Function': 'GravitySource',
    #    "gravity": 9.8,
    #    "source_treatment": "Explicit",
    #},
    #'source3': {
        #'Function': 'ExsolutionSource',   # Exsolution source
        #'source_treatment': 'Explicit',
        #'tau_d': 10.0,                    # Exsolution timescale (s)
    #},
    #'source4': {'Function': 'FragmentationTimescaleSourceSmoothed',
    #    'crit_volfrac': 0.7,                       # Critical volume fraction
    #    'fragsmooth_scale': 0.05,                  # Fragmentation smoothing scale (for two-sided smoothing)
    #    'source_treatment': 'Explicit',
    #    'tau_f': 1.0,                              # Fragmentation timescale (s)
    #},
    ## Add a new source term that does nothing yet; see src/physics/multiphasevpT/functions.py
    ## It is easy to modify this source term to affect the value of the new state variable as a function of x and t. 
    "slip_source": {
        "Function" : "SlipSource",
        "source_treatment" : "Explicit",
    },
    "conduit_wall_drag": {
        "Function": "FrictionVolSlip",
        "source_treatment": "Explicit",
        "conduit_radius": 10,    # Condut radius[m]
        "tau_p": 2e5,            # Primary shear stress from slip [Pa]
        "tau_r": 0,            # Residual shear stress from slip [Pa]
        "D_c": 3.5,             #[m]
        "plug_boundary_0" : -100,
        "use_constant_tau": True,
        "dissipate_shear_heat": True,  # If true, model the shear heat dissipation through conduction in the conduit. 
        "exponential_friction": False,  # If true, use an exponential friction law to calculate tau.
    },
    # Add existing source terms (turn off by removing item from the SourceTerms dictionary)
    'viscous_drag': {
        'Function': 'FrictionVolFracVariableMu',   # Friction source term for given conduit radius
        'conduit_radius': 10,
        'use_default_viscosity': True,
        'default_viscosity': 2e6,
        'source_treatment': 'Explicit',
        'plug_boundary_0': -100,
        'dissipate_heat': True,  # If true, dissipate the heat from viscous drag through the conduit walls. 
        'model_plug': True,
    },
}

# An "exact solution" is needed by Quail, but does not need to be called
# This is a random function used as a placeholder
ExactSolution = {
    "Function": "RiemannProblem",
}

LinkedSolvers = []

BoundaryConditions = {
    'x1': {
        'BCType': 'SlipWall',   # Inlet boundary condition
    },
    #'x1': {
    #    'BCType': 'VelocityInlet1D',   # Inlet boundary condition
    #    "u": 0.4,                      # Velocity
    #    "yWt": 0.0,                    # Mass total water (exsolved + dissolved) per mixture volume
    #    "p_chamber": 10e6,
    #},
    "x2" : {
        "BCType" : "PressureOutlet1D",
        "p": 1e5,
    },
}