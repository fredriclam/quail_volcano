import numpy as np
from physics.multiphasevpT.hydrostatic1D import GlobalDG

Numerics = {
    "SolutionOrder" : 1,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    # "ApplyLimiters" : ["WENO", "PositivityPreservingMultiphasevpT"],
    # "ApplyLimiters" : ["WENO"],
    # "ShockIndicator": "MinMod", "TVBParameter": 0.2,
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
        # Flag to use artificial viscosity
		# If true, artificial visocity will be added
    "ArtificialViscosity" : True,
	"AVParameter" : 30, # 150 ~ 500 is ok for this commit #150, #50, #1e-5, #1e3, 5e3,
    'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "temp_source_conduit1_test",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 120, #3*301, #151,#351, # Use even number if using p_jump IC
    "xmin" : -1200.0-150, # With shift (-150 m)
    "xmax" : 0.0-150,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

# Location of jump (sync both initial condition to specify jump in fluid
# composition, and in hydrostatic_solve to specify jump in pressure)
x_jump = -550.0

InitialCondition = {
	# "Function" : "UniformExsolutionTest",
    "Function" : "RiemannProblem",
    "xd": x_jump,
    # "rhoL": 12.5,
    # "uL": 0.0,
    # "pL": 10*1e5,
    # "rhoR": 1.25,
    # "uR": 0.0,
    # "pR": 1e5,
    # arhoAL=1., arhoWvL=1., arhoML=2e5, uL=0., TL=1000., 
	            #  arhoAR=10., arhoWvR=0., arhoMR=0.125, uR=0., TR=300., xd=0.)
    "arhoAL": 1e-3, # 4wt%, 5 MPa water design
    "arhoWvL": 9.874849008229967e+00, # 9.474849008229967e+00,
    "arhoML": 3.142648542057735e+02, # Near critical, plenty of initial exsolved
	"TL": 1000.,
    "arhoWtL": 1.694958812856014e+01, #1.294958812856014e+01,
}

# List of functions to inject in custom user function
def hydrostatic_solve(solver, owner_domain=None):
    GlobalDG(solver).set_initial_condition(
        p_bdry=1e5,
        is_jump_included=True,
        owner_domain=owner_domain,
        constr_key="YEq",
        x_jump=x_jump,
    )

Inject = [
    {
        "Function": hydrostatic_solve,
        "Initial": True,
        "Postinitial": False,
    }
]

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
        "source_treatment" : "Explicit",
	},
    "source2": {
        "Function": "FrictionVolFracConstMu",
        "source_treatment" : "Explicit",
        "crit_volfrac": 0.6,
    },
    "source3": {
        "Function": "ExsolutionSource",
        "source_treatment" : "Explicit",
    },
    "source4": {
        "Function": "FragmentationTimescaleSource",
        "source_treatment": "Explicit",
    }
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

extend_conduit = False # TODO: change back
if extend_conduit:
    BoundaryConditions = {
        "x1" : {
            "BCType" : "MultiphasevpT1D1D",
            "bkey": "interface_-1",
        },
        "x2" : { 
            "BCType" : "MultiphasevpT2D1D", # TODO: implement r-weighted integration
            "bkey": "vent",
        },
    }

    LinkedSolvers = [
        {
            "DeckName": "conduit2.py",
            "BoundaryName": "interface_-1",
        },
    ]
else:
    BoundaryConditions = {
        "x1" : {
            "BCType" : "SlipWall",
        },
        # "x2" : { 
        #     "BCType" : "SlipWall", # TODO: implement r-weighted integration (check)
        #     # "bkey": "vent",
        # },
        "x2" : { 
            "BCType" : "MultiphasevpT2D1D", # TODO: implement r-weighted integration
            "bkey": "vent",
        },
    }