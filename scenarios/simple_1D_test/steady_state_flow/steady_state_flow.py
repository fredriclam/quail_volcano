
TimeStepping = {
	"InitialTime"  : 0.0,
	"FinalTime"    : 30,      # seconds
	"NumTimeSteps" : 6000,
	"TimeStepper"  : "RK3SR",
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
	"Prefix" : "test_variable_tau",
	"WriteInterval" : 60,
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
    "Liquid": {"K": 10e9,     # Condensed phase bulk modulus (Pa)
               "rho0": 2.6e3, # Condensed phase density (kg/m^3)
               "p0": 5e6,     # Condensed phase EOS reference pressure (Pa)
               "E_m0": 0,     # Leave zero
               "c_m": 1e3},   # Condensed phase heat capacity (J / (kg K))
}

# Set a spatially uniform initial condition
# Here we specify the state variables corresponding to mass,
# and velocity u and temperature T. From velocity u and the
# mass variables, momentum is calculated internally. The total
# energy is calculated internally from temperature and all
# other provided variables.
InitialCondition = {
    "Function": "UniformTest",
    "arhoA": 0.0,    # Mass air per mixture volume
    "arhoWv": 0.001,   # Mass water in exsolved state
    "arhoM": 2550.0, # Mass condensed phase per mixture volume
    "u": 0.0,        # Velocity
    "T": 1000.0,     # Temperature
    "arhoWt": 0.001, # Mass total water (exsolved + dissolved) per mixture volume
    "arhoC": 0.0,        # Mass crystals per mixture volume
    "arhoF": 0.0,          # Mass fragmented magma per mixture volume
    "arhoS": 0.0,          # Newly implemented state
}

SourceTerms = {
    # Add existing source terms (turn off by removing item from the SourceTerms dictionary)
    'source1': {
        'Function': 'FrictionVolFracVariableMu',   # Friction source term for given conduit radius
        'conduit_radius': 50,
        'use_default_viscosity': True,
        'default_viscosity': 5e5,
        'source_treatment': 'Explicit'
    },
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
    "source5": {
        "Function" : "SlipSource",
        "source_treatment" : "Explicit",
    },
    "source6": {
        "Function": "FrictionVolSlip",
        "source_treatment": "Explicit",
        "tau_p": 2e5,       #[Pa]
        "tau_r": 5e4,       #[Pa]
        "D_c": 10,          #[m]
    }
}

# An "exact solution" is needed by Quail, but does not need to be called
# This is a random function used as a placeholder
ExactSolution = {
    "Function": "RiemannProblem",
}

LinkedSolvers = []

BoundaryConditions = {
    'x1': {
        'BCType': 'PressureStableLinearizedInlet1D',   # Inlet boundary condition
        'p_chamber': 1e7, # Chamber pressure (Pa)
        'T_chamber': 1000, # Chamber temperature (K)
        'trace_arho': 0,
        'chi_water': 0,
        'cVFav': 0,
        'arhoS': 2,  # Newly implemented state
    },
    "x2" : {
        "BCType" : "PressureOutlet1D",
        "p": 1e6,
    },
}