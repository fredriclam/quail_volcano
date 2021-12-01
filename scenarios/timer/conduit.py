import numpy as np
import copy

Numerics = {
    "SolutionOrder" : 0,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    # "ApplyLimiters" : "PositivityPreserving",
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
}

Output = {
	"Prefix" : "debug_standard3",
	"WriteInterval" : 100,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 1500,#800,
    "xmin" : -153.0,#-23.0,
    "xmax" : -3.0,
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 287.,
    "SpecificHeatRatio" : 1.4,
}

SourceTerms = {
    # "source1" : {
    #     "Function" : "PorousSource",
    #     "nu": 0.0*-.5,
    #     "alpha": 0.0*2.0e4,
    #     "T_m": 400.0,
    # },
}

rhoAmbient = 3.0 # 1.2
TAmbient = 800.0 # 300.0
eAmbient = rhoAmbient * Physics["GasConstant"] / (Physics["SpecificHeatRatio"] - 1.0) * TAmbient
pAmbient = rhoAmbient * Physics["GasConstant"] * TAmbient

if False:
    # Sod state
    rhoAmbient = 1.0 # 1.2
    # TAmbient = 400.0 # 300.0
    pAmbient = 1.0
    eAmbient = pAmbient / (Physics["SpecificHeatRatio"] - 1.0)

UQuiescent = np.array([rhoAmbient, 0.0, eAmbient])

InitialCondition = {
	"Function" : "Uniform",
	"state" : UQuiescent,
}

if False:
    InitialCondition = {
        "Function" : "MultipleRiemann",
        # "Function" : "Uniform",
        # "state" : UQuiescent,
        "rhoL": 1.0,
        "uL": 0.0,
        "pL": 1.0,
        "rhoR": 0.125,
        "uR": 0.0,
        "pR": 0.1,
    }

# Fake exact solution
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
  "x1" : {
      "BCType" : "SlipWall",
  },
  "x2" : { 
      "BCType" : "Euler2D1D",
      "bkey": "x2",
  },
}