import numpy as np
from physics.multiphasevpT.hydrostatic1D import GlobalDG
from pyXSteam.XSteam import XSteam
import scipy

# Set timestepper
TimeStepping = {
	"InitialTime" : 0.0,
	"FinalTime" : 2.50,
	"NumTimeSteps" : 25000,
  # TimeStepper options:
  # FE, SSPRK3, RK4, Strang (split for implicit source treatment)
	"TimeStepper" : "FE",
}

Numerics = {
  # Solution order; these correspond to:
  # 0: 1 node in each element representing the average value
  # 1: 2 nodes in each element constituting a linear representation of the
  #    solution in the element
  # 2: 3 nodes in each element constituting a quadratic representation of the
  #    solution in the element
  # Order 0 is most robust, and the error is mostly diffusive (may be
  # currently bugged, as of commit 68ce3b653480e52a1d52436f0206b3327c8f744f
  # on branch main.
  "SolutionOrder" : 0,
  "SolutionBasis" : "LagrangeSeg",
  "Solver" : "DG",
  # "ApplyLimiters" : "PositivityPreservingMultiphasevpT",
  # The following is a way to stack additional limiters (WENO limiter is
  # designed for the Euler equations and reduces oscillations while preserving
  # order of accuracy)
  # "ApplyLimiters" : ["WENO", "PositivityPreservingMultiphasevpT"],
  #   "ShockIndicator": "MinMod", "TVBParameter": 0.2,
  "ElementQuadrature" : "GaussLegendre",
  "FaceQuadrature" : "GaussLegendre",
  # Artificial viscosity adds a diffusion term to all equations, proportional to
  #   |grad p| / p + |grad (alpha_a * rho_a)| / (alpha_a * rho_a),
  # the cube of the element size (h^3), and AVParameter.
  "ArtificialViscosity" : False,
  "AVParameter" : 500,
  'L2InitialCondition': False, # If false, use interpolation instead of L2 projection of Riemann data
}

Output = {
	"Prefix" : "conduit1D",
  # Write to disk every WriteInterval timesteps
	"WriteInterval" : 30,
	"WriteInitialSolution" : True,
  # Automatically queues up post_process.py after this file (see Quail examples)
	"AutoPostProcess": False,
}

Mesh = {
    "File" : None,
    "ElementShape" : "Segment",
    # Use even number if using initial condition with discontinuous pressure
    "NumElemsX" : 60,
    "xmin" : -1200.0,
    "xmax" : 0.0,
}

Physics = {
    "Type" : "MultiphasevpT",
    "ConvFluxNumerical" : "LaxFriedrichs",
}

InitialCondition = {
  # Initial condition (not necessarily hydrostatic; the injected function (see
 # below) takes this initial condition and computes the hydrostatic solution).
 # The parameters can be provided below, or changed in the source:
 # quail_volcano\src\physics\multiphasevpT\functions.py > RiemannProblem
 "Function" : "RiemannProblem",
 # The following are optional parameters. If not provided, the default args
 # are used.
 # Left side values
"arhoAL": 0.,
"arhoWvL": 5,
"arhoML": 2496.3,
"uL": 0.,
"TL": 800.,
"arhoWtL": 0.,
"arhoCL": 0.,
# Right side values
"arhoAR": 0.,
"arhoWvR": 5,
"arhoMR": 2496.3,
"uR": 0.,
"TR": 800.,
"arhoWtR": 0.,
"arhoCR": 0.0,
#"xd": -600.0,  # Position of the discontinuity
}



# Define the hydrostatic steady-state solver that operates on the initial
# condition provided. The solver solves the discontinuous Galerkin steady-state
# problem, and preserves certain properties of the provided initial condition.
# See hydrostatic1D.py for more details
def hydrostatic_solve(solver, owner_domain=None):
    GlobalDG(solver).set_initial_condition(
        p_bdry=1e5,  # boundary pressure, started at 1e5
        is_jump_included=False,
        owner_domain=owner_domain,
        x_jump=-600.0,
        constr_key="YEq",
        # To set the traction function, use the following line and prescribe
        # traction as a function of x. The traction function needs to be
        # somehow compatible with the initial condition in RiemannProblem.
        # Maybe even a smoothed out version of Riemann problem would be needed.
        # traction_fn=lambda x: (-1e7/(50.0*np.sqrt(np.pi)))*np.exp(-((x-0.0)/50.0)**2.0)
    )

# Inject is a list of dicts with a value "Function" pointing to a function
# provided args (solver, (optional: owner_domain)). These functions are injected
# into Quail to make them run before the first timestep, and once after each
# timestep. If Initial is provided True, the function runs before the first timestep.
# If Postinitial is provided True, the function runs after each timestep.
def update_pressure_temperature(solver, owner_domain=None):
    # steam table initialized
    steam_table = XSteam(XSteam.UNIT_SYSTEM_BARE)

    def create_functions(variables, other_states):
        (pressure, enthalpy) = variables
        (arhoWv_val, arhoM_val, e_val, mom_val) = other_states

        # volume fraction of water vapor
        alpha_w = (solver.physics.Liquid["p0"] - solver.physics.Liquid["K"] - 1e6 * pressure +
                   (solver.physics.Liquid["K"] / solver.physics.Liquid["rho0"]) * arhoM_val) / \
                  (solver.physics.Liquid["p0"] - solver.physics.Liquid["K"] - 1e6 * pressure)
        internal_energy_magma = arhoM_val * solver.physics.Liquid["c_m"] * 1e-3 * steam_table.t_ph(pressure, enthalpy)

        # if np.isnan(steam_table.u_pt(pressure, temperature)) or np.isnan(steam_table.h_pt(pressure, temperature)):
        #     # optimize over x_frac and pressure --> extract pressure and temperature
        #     print("Hello")

        # eqn for energy conservation
        eqn_1 = (mom_val ** 2 / (2 * (arhoM_val + arhoWv_val))) * 1e-3 \
                + arhoWv_val * steam_table.u_ph(pressure,
                                                enthalpy) + internal_energy_magma - e_val * 1e-3

        # eqn of state relating enthalpy, pressure, and energy
        eqn_2 = -enthalpy + steam_table.u_ph(pressure, enthalpy) \
                + (pressure * 1e3 * alpha_w) / arhoWv_val
        return np.array([eqn_1, eqn_2/1e4]).ravel()

    Uq = solver.state_coeffs
    ''' Extract state variables '''
    arhoWv = Uq[:, :, solver.physics.get_state_slice("pDensityWv")]
    arhoM = Uq[:, :, solver.physics.get_state_slice("pDensityM")]
    mom = Uq[:, :, solver.physics.get_momentum_slice()]
    e = Uq[:, :, solver.physics.get_state_slice("Energy")]
    arhoA = Uq[:, :, solver.physics.get_state_slice("pDensityA")]

    # ''' Flag non-physical state '''
    # if flag_non_physical:
    #     if np.any(arhoA < 0.) or np.any(arhoWv < 0.) or np.any(arhoM < 0.) \
    #             or np.any(arhoWt < 0.):  # or np.any(arhoC < 0.):
    #         raise errors.NotPhysicalError

    ub = (65., 3850.)  # upper bound: (pressure, enthalpy)
    lb = (0.000611213, 400.)  # lower bound -- 611.213 Pa (the lb for pressure) is in IAPWS documentation for lowest pressure
    solution = np.zeros(shape=(Uq.shape[0], 7), dtype=float)
    previous_x0 = (-1, -1)
    for i in range(Uq.shape[0] - 1, -1, -1):
        if previous_x0 == (-1, -1):
            x0 = (60., 3000.)
        else:
            #x0 = previous_x0  # best guess is last solution
            x0 = (60., 3000.)
        # if np.isnan(steam_table.u_pt(x0[0], x0[1])) or np.isnan(steam_table.h_pt(x0[0], x0[1])):
        #     print("Hello")
        other_variables = (arhoWv[i, 0, 0], arhoM[i, 0, 0], e[i, 0, 0], mom[i, 0, 0])
        solution_val = scipy.optimize.least_squares(lambda variables: create_functions(variables, other_variables),
                                                   x0, bounds=(lb, ub), ftol=1e-6, xtol=1e-6, gtol=1e-6)

        if np.linalg.norm(solution_val["fun"]) > 1e2:
            print("ERROR! TOO BIG!! WRONG VALUE")
            print(solution_val["fun"])
        solution[i, 0] = solution_val["x"][0] * 1e6  # pressure
        solution[i, 1] = solution_val["x"][1]  # enthalpy
        previous_x0 = (solution_val["x"][0], solution_val["x"][1])

    # save other variables so that you can find them for single value case
    solution[:, 2] = arhoWv.squeeze()
    solution[:, 3] = arhoM.squeeze()
    solution[:, 4] = e.squeeze()
    solution[:, 5] = arhoA.squeeze()
    solution[0, 6] = solver.time

    # store in new variable
    solver.physics.pressure_temp = solution

# changed the inject fuction so that it pre-runs the optimization of pressure and enthalpy, so it doesn't have to be
# computed multiple times a step
Inject = [
    {
        "Function": update_pressure_temperature,
        "Initial": True,
        "Postinitial": True,
    }
]

# Add source terms here. Source terms just stack up, and can be named whatever
# is convenient. source_treatment is only relevant for splitting schemes, like
# "Strang" in TimeStepping options.
SourceTerms = {
	# "source1": {
	# 	"Function" : "GravitySource",
	# 	"gravity": 9.8,
    # "source_treatment" : "Explicit",
	# },
  # "source2": {
  #     "Function": "FrictionVolFracConstMu",
  #     "source_treatment" : "Explicit",
      # Some options, and their default values
      # "mu": 1e5,
      # "conduit_radius": 50.0,
			# "crit_volfrac": 0.8,
      # "logistic_scale": 0.01,
  # },
  # "source3": {
  #     "Function": "ExsolutionSource",
  #     "source_treatment" : "Implicit",
  #     "tau_d": 1.0,
  # },
 "source4": {
     "Function" : "WaterInflowSource",
     "aquifer_depth" : -540,  # the aquifer depth is difference from the deepest part of the conduit
     "aquifer_length" : 120,  # the size of the conduit
 },
}

# Fake exact solution
ExactSolution = InitialCondition.copy()

BoundaryConditions = {
    # The leftmost boundary
    "x1" : {
      # To be replaced by an exit pressure boundary condition
      "BCType" : "SlipWall"
      # To use multiple domains (for parallelism), the below can be uncommented
      # and bkey set to a name that is known to this solver and a linked solver.
      # See LinkedSolvers below for parallelism
      # "BCType" : "MultiphasevpT1D1D",
      # "bkey": "interface_-1",
    },
    "x2" : { 
        "BCType" : "SlipWall",
    },
}

# The solvers/domains that are linked to this one through a coupling BC.
# DeckName refers to the name of a copy of this file, specifying the parameters
# for the solver on the linked domain. The linked domain does not need to link
# back to this one, and does not need a TimeStepping option (TimeStepping
# options are taken from this root parameter file).
# The BoundaryName is the name given to the coupling boundary condition. It
# should be unique across all linked solvers, and named in the bkey of the
# corresponding BoundaryCondition (for example, for boundary "x1" here and 
# boundary "x2" in the linked parameter file).
LinkedSolvers = [
    # {
    #     "DeckName": "conduit2.py",
    #     "BoundaryName": "interface_-1",
    # },
]
