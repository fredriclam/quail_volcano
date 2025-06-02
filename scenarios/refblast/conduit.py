import numpy as np
import run_globals

Numerics = {
    "SolutionOrder" : 1,
    "SolutionBasis" : "LagrangeSeg",
    "Solver" : "DG",
    "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
    "ElementQuadrature" : "GaussLegendre",
    "FaceQuadrature" : "GaussLegendre",
    "ArtificialViscosity" : True,
	"AVParameter" : 30,
    'L2InitialCondition': False,
}

Output = {
	"Prefix" : f"{run_globals.file_prefix}_cond",
	"WriteInterval" : run_globals.write_interval,
	"WriteInitialSolution" : True,
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    "NumElemsX" : 800,
    "xmin" : -2000.0 - 150, # With shift (-150 m)
    "xmax" : 0.0 - 150,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

def smoother(x, scale):
    # Shift, scale, and clip to [-1, 0] to prevent exp overflow
    _x = np.clip(x / scale + 1, 0, 1)
    f0 = np.exp(-1/np.where(_x == 0, 1, _x))
    f1 = np.exp(-1/np.where(_x == 1, 1, 1-_x))
    # Return piecewise evaluation
    return np.where(_x >= 1, 1,
                    np.where(_x <= 0, 0, 
                        f0 / (f0 + f1)))

def bump(x, a, b, scale):
    ''' Smoothed bump function with exterior smoothing '''
    return smoother(x - a, scale) * smoother(b - x, scale)

p_chamber = 45e6
T_chamber = 1273.15
yWt = 0.02
yC = 0.15

def traction_fn(x):
    return -bump(x, -550, -450, 20) #-bump(x, -250, -150, 20)
def yWt_fn(x):
    return yWt * np.ones_like(x)
def yC_fn(x):
    return yC * np.ones_like(x)
def T_fn(x):
    return T_chamber * np.ones_like(x)


InitialCondition = {
    "Function": "StaticPlug",
    "traction_fn": traction_fn,
    "yWt_fn": yWt_fn,
    "yC_fn": yC_fn,
    "T_fn": T_fn,
    "x_global": np.linspace(Mesh["xmin"], Mesh["xmax"], Mesh["NumElemsX"]),
    "p_chamber": p_chamber,
    "yA": 1e-5,
    "c_v_magma": 1e3,
    "rho0_magma": 2.7e3,
    "K_magma": 10e9,
    "p0_magma": 5e6,
    "solubility_k": 5e-6,
    "solubility_n": 0.5,
    "neglect_edfm": True,
    "enforce_p_vent": 1.01325e5,
}

SourceTerms = {
	"source1": {
		"Function" : "GravitySource",
		"gravity": 9.8,
        "source_treatment" : "Explicit",
	},
    "source2": {
        "Function": "FrictionVolFracVariableMu",
        "source_treatment" : "Explicit",
        "viscosity_factor": 1.0,
        "conduit_radius": 50.0,
    },
    "source3": {
        "Function": "ExsolutionSource",
        "source_treatment" : "Explicit",
    },
    "source5": {
        "Function": "FragmentationStrainRateSource",
        "tau_f": 0.005,
        "G": 1e9,
        "mu0": 1e9,
        "k": 0.001,
        "fragsmooth_scale": 0.010,
        "which_criterion": "both",
        "conduit_radius": 50,
    },
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

chi_water = yWt / (1 - (yC + yWt))
BoundaryConditions = {
    "x1" : {
        "BCType" : "PressureStableLinearizedInlet1D",
        "p_chamber": p_chamber,
        "T_chamber": T_chamber,
        "trace_arho": 1e-6,
        "chi_water": chi_water,
        "cVFav": yC,
        "cVFamp": 0.0,
        "is_gaussian": True,
    },
    "x2" : { 
        "BCType" : "MultiphasevpT2D1D",
        "bkey": "x2",
    },
}
