import numpy as np
from physics.multiphasevpT.hydrostatic1D import GlobalDG

TimeStepping = {
    "InitialTime" : 0.0,
    "FinalTime" : 5,
    "NumTimeSteps" : 10000,
    "TimeStepper" : "FE",
}

Numerics = {
    "SolutionOrder" : 1,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    # "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "ArtificialViscosity" : False,
	"AVParameter" : 150,
    'L2InitialCondition': True, # L2InitialCondition ? (L2 projection, Interpolation)
}

Output = {
	"Prefix" : "verification_conduit_p",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 100,
    "xmin" : 0.0,
    "xmax" : 1000.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
    "Function" : "RightTravelingGaussian",
    "amplitude": 1,
    "location": 800,
}

# List of functions to inject in custom user function
def hydrostatic_solve(solver, owner_domain=None):
    GlobalDG(solver).set_initial_condition(
        p_bdry=1e5,
        is_jump_included=True,
        owner_domain=owner_domain,
        # traction_fn=lambda x: (-1e7/(50.0*np.sqrt(np.pi)))*np.exp(-((x-0.0)/50.0)**2.0)
    )

Inject = [
    # {
    #     "Function": hydrostatic_solve,
    #     "Initial": True,
    #     "Postinitial": False,
    # }
]

SourceTerms = {
	# "source1": {
	# 	"Function" : "GravitySource",
	# 	"gravity": 9.8,
    #     "source_treatment" : "Explicit",
	# },
    # "source2": {
    #     "Function": "FrictionVolFracConstMu",
    #     "source_treatment" : "Explicit",
    # },
    # "source3": {
    #     "Function": "ExsolutionSource",
    #     "source_treatment" : "Implicit",
    # },
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

IsDecoupled = True
if IsDecoupled:
    BoundaryConditions = {
        "x1" : {
            "BCType" : "SlipWall"
        },
        "x2" : { 
            "BCType" : "PressureOutlet1D",
            "p": 1e5,
        },
    }
else:
    BoundaryConditions = {
        "x1" : {
            # "BCType" : "SlipWall"
            "BCType" : "MultiphasevpT1D1D",
            "bkey": "interface_-1",
        },
        "x2" : { 
            # "BCType" : "SlipWall",
            "BCType" : "MultiphasevpT2D1D",
            "bkey": "vent",
        },
    }

    LinkedSolvers = [
        {
            "DeckName": "conduit2.py",
            "BoundaryName": "interface_-1",
        },
    ]