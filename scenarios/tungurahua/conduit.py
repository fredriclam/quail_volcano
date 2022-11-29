import numpy as np
from physics.multiphasevpT.hydrostatic1D import GlobalDG

Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
        # Flag to use artificial viscosity
		# If true, artificial visocity will be added
    "ArtificialViscosity" : True,
	"AVParameter" : 150, # 150 ~ 500 is ok for this commit #150, #50, #1e-5, #1e3, 5e3,
    'L2InitialCondition': False, # Use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "tung3_conduit1",
	"WriteInterval" : 400,
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
    },
    "source3": {
        "Function": "ExsolutionSource",
        "source_treatment" : "Explicit",
    },
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

extend_conduit = True
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
        "x2" : { 
            "BCType" : "MultiphasevpT2D1D", # TODO: implement r-weighted integration
            "bkey": "vent",
        },
    }