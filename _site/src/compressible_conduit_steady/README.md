# Steady-state compressible flow for volcanic conduit modeling

This repository contains the code for steady-state, quasi-1D volcanic conduit modeling for a compressible, multiphase mixture. This code solves the system of ordinary differential equations (ODE) that results from the steady-state assumption to the model in [quail_volcano](https://github.com/fredriclam/quail_volcano/tree/main), and is used as a submodule in quail_volcano for setting steady-state initial conditions there.

The multiphase mixture consists of air, water, and magma at a single mixture pressure, temperature, and velocity.

The built-in ODE solver of python's `scipy` package is used to solve the ODE.

This code is used in the following (forthcoming) works:

* Coppess, K. R., Lam, F. Y. K., & Dunham, E. M. Seismic signatures of fluctuating fragmentation in volcanic eruptions. (in prep)
* Coppess, K. R., Lam, F. Y. K., & Dunham, E. M. Volcanic eruption tremor from  particle impacts and turbulence using conduit flow models. (submitted to *Seismica*)

# License

This code is open-source under the MIT license.

# Setup

Python 3.9 is recommended. The latest version of `numpy` and `scipy` are recommended. Clone repository.

# Basic usage

For an example of basic usage of the steady-state solver, such as for solving the hydrostatic problem (zero velocity), see [startup_hydrostatic_only.ipynb](https://github.com/fredriclam/compressible-conduit-steady/blob/main/startup_hydrostatic_only.ipynb), replacing in the first box `source_dir` with your working directory. This notebook solves the hydrostatic problem, with the top boundary held at atmospheric pressure. The output format is in terms of the 1-D conserved variables used in [quail_volcano](https://github.com/fredriclam/quail_volcano/tree/main). To obtain the output in terms of pressure, change the flag `io_format` as indicated in the comments in the notebook.
