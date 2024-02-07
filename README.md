This repository contains the code for 1D/2D explosive volcanic eruptions simulation, extending the open-source discontinuous Galerkin prototyping framework [Quail](https://github.com/IhmeGroup/quail) [1], developed by the [FxLab at Stanford](https://github.com/IhmeGroup). The code is set up to run unsteady, compressible, multiphase flow, consisting of air, water, and magma. The implementation allows coupled 1-D, 2-D, and 2-D axisymmetric domains, run on separate threads.

1. [Ching, E. J., Bornhoft, B., Lasemi, A., & Ihme, M. (2022). Quail: A lightweight open-source discontinuous Galerkin code in Python for teaching and prototyping. SoftwareX, 17, 100982.](https://doi.org/10.1016/j.softx.2022.100982)

This code will appear in the following forthcoming works:

* Coppess, K. R., Lam, F. Y. K., & Dunham, E. M. Seismic signatures of fluctuating fragmentation in volcanic eruptions. (in prep)
* Lam, F. Y. K., & Dunham, E. M. A multiphase model for explosive volcanic eruptions using the discontinuous Galerkin method. (in prep)

### Seismic signatures

The code used to compute seismic signatures of an unsteady eruption can be found at [scenarios/conduit_flow](https://github.com/fredriclam/quail_volcano/tree/main/scenarios/conduit_flow).

### License

This code is open-source under [GPL-3.0](https://github.com/fredriclam/quail_volcano/blob/main/LICENSE).

### See also

The 1-D steady-state implementation of the same governing equations is available on github: [compressible-conduit-steady](https://github.com/fredriclam/compressible-conduit-steady/tree/main). The use of the steady-state solver, which is an independent implementation using python's built-in ODE solvers, is appropriate for long timescale simulations.

### Setup

Python 3.9 is recommended. The latest version of `numpy` and `scipy` are recommended. See also the [Quail documentation](https://github.com/IhmeGroup/quail) for additional help setting up the code.

With python, numpy and scipy installed, clone this repository. Then, install submodules (including the steady-solver for setting up the initial condition) by using
```
git submodule init compressible-conduit-steady
git submodule update
```
to load the submodule. To update the local code on your computer, use
```
git pull origin main
```
to update the main branch. Furthermore, to update the submodule, navigate to the submodule (e.g., `src/compressible_conduit_steady`)and use the same command.
