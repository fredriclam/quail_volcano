''' Sod shock tube problem for the multiphase mixture. '''

TimeStepping = {
	"InitialTime"  : 0.0,
	"FinalTime"    : 1.0,      # seconds
	"NumTimeSteps" : 1000,
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
	"AVParameter" : 30, #0.3,
    'L2InitialCondition': True,
}

Output = {
	"Prefix" : "test_output",
	"WriteInterval" : 10,
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
    "arhoWv": 0.8,   # Mass water in fluid state per mixture volume
    "arhoM": 2500.0, # Mass condensed phase per mixture volume
    "u": 0.0,        # Velocity
    "T": 1000.0,     # Temperature
    "arhoWt": 2500.0*0.04, # Mass total water (fluid + dissolved) per mixture volume
    "arhoC": 100.0,        # Mass crystals per mixture volume
    "arhoF": 0.0,          # Mass fragmented magma per mixture volume
    "arhoX": 0.0,          # Newly implemented state
}

SourceTerms = {}

# An "exact solution" is needed by Quail, but does not need to be called
# This is a random function used as a placeholder
ExactSolution = {
    "Function": "RiemannProblem",
}

LinkedSolvers = []

BoundaryConditions = {
    "x1" : {
        "BCType" : "SlipWall",
    },
    "x2" : {
        "BCType" : "SlipWall",
    },
}