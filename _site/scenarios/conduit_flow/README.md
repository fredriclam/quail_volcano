This folder contains the code used in "Seismic signatures of fluctuating fragmentation in volcanic eruptions."

The instructions to run a particular simulation are as follows:

0. If running the code on a cluster, load the python3, numpy and scipy modules.
1. Navigate to this folder on your computer (`.../quail_volcano/scenarios/conduit_flow`), where `...` is the folder where quail_volcano is located.
2. Identify the main parameter file needed for the simulation. One parameter file is provided for each 1-D or 2-D domain in the simulation (for example, the volcanic conduit may be a 1-D domain connected to a 2-D axisymmetric atmosphere). The main file controls the timestepping parameters, and recursively identifies the other domains in the simulation. The parameter file names are provided in the section below.
3. Call quail and provide this main parameter file:
   ```
   python3 .../quail_volcano/src/quail yourparameterfilename
   ```
   where `...` is the folder where quail_volcano is located, and `yourparameterfilename` is to be replaced by the parameter file identified in step 2.
4. The output will be in python binaries (`.pkl`) that can be post-processed as in the provided IPython (`.ipynb`) notebooks. These `.pkl` files are saved at fixed intervals, as specified in the corresponding parameter file.

### Select simulations

The parameter file names to be provided to quail are as follows:

