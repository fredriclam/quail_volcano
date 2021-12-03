import numpy as np
import copy

Numerics = {
    "SolutionOrder" : 1,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    # "ApplyLimiters" : "PositivityPreserving",
    # "NodeType" : "Equidistant",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "ArtificialViscosity" : True,
		# Flag to use artificial viscosity
		# If true, artificial visocity will be added
	"AVParameter" : 1e4,#5e3,
		# Parameter in the artificial viscosity term. A larger value will
		# increase the amount of AV added, giving a smoother solution.
}

Output = {
	"Prefix" : "referenceD_conduit",
	"WriteInterval" : 200,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 1000,
    "xmin" : -5000.0-150.0,
    "xmax" : -150.0,
}

Physics = {
    "Type" : "Euler",
    "ConvFluxNumerical" : "Roe",
    "GasConstant" : 287.,
    "SpecificHeatRatio" : 1.4,
}

SourceTerms = {
    "source1" : {
        "Function" : "PorousSource",
        "nu": 0.0*-.5,
        "alpha": 0.0*2.0e4,
        "T_m": 400.0,
    },
}

pAmbient = 100*1e5
TAmbient = 1000.0 # 300.0
rhoAmbient = pAmbient / ( Physics["GasConstant"] * TAmbient )

eAmbient = rhoAmbient * Physics["GasConstant"] / (Physics["SpecificHeatRatio"] - 1.0) * TAmbient
# rhoAmbient = 3.0 # 1.2
# pAmbient = rhoAmbient * Physics["GasConstant"] * TAmbient

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