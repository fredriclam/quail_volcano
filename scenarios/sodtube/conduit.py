''' Sod shock tube problem for the multiphase mixture. '''

TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 1,
    # NumTimeSteps CFL: scales with NumElemsX as
    #   ~ Nx/2 * (2p+1) * (|M| + 1)
    #   A posteriori |M| <= 2.7
    #   Then 1/2 * (2p+1) * 3.7 = {P0:1.85, P1:5.55, P2:9.25}
    # Note that RK3SR is more efficient than FE, so in practice the CFL
    # restriction is a bit more lax.
	"NumTimeSteps" : 1600,
	"TimeStepper" : "RK3SR",
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
	"Prefix" : "sodtube_AVnonhydrostatic_p0",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    # Use even number if using initial condition with discontinuous pressure
    "NumElemsX" : 200,
    "xmin" : -2600,
    "xmax" : 2600,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
    "Liquid": {"K": 10e9,
               "rho0": 2.7e3,
               "p0": 5e6,
               "E_m0": 0,
               "c_m": 3e3},
}

InitialCondition = {
    "Function": "RiemannProblem",
    "arhoAL": 1e-1,
    "arhoWvL": 8.686,
    "arhoML": 2496.3,
    "uL": 0.0,
    "TL": 1000,
    "arhoWtL": 10.0,
    "arhoCL": 1e-2,
    "arhoFmL": 1e-2,
    "arhoAR": 1.161,
    "arhoWvR": 1.161*5e-3,
    "arhoMR": 1e-5,
    "uR": 0.0,
    "TR": 300,
    "arhoWtR": 1.161*5e-3,
    "arhoCR": 0.5e-5,
    "arhoFmR": 0.5e-5,
    "xd": 0.0,
}

SourceTerms = {}

# Fake exact solution
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