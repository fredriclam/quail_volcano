### Plotting tools

Plotting tools are found in `.\src\processing\mdtools.py`.

`mdtools` contains tools for plotting multidomain objects. The following functions are provided for convenient plotting. Probably the useful one is `viz`.

`viz(solver, plot_qty="Pressure", levels=None, clims=None)`: quick visualization of solver. String `plot_qty` is the name of the quantity defined in multiphasevpT.py (search term `AdditionalVariables`; these are the variable names like `Pressure`). The names of the conserved variables also work: these are `pDensityA`, `pDensityWv`, `pDensityM`, `XMomentum` (and `YMomentum`), `Energy`, `pDensityWt`, and `pDensityC`. This works for 1D and 2D.

`downsample(solver, plot_qty:str="Pressure")`: returns `(x, var_plot)`, a downsample of the finite element solution suitable for plotting. Downsampling is effective for obtaining a coarse mesh of points for quickly plotting arbitrary solutions. Variable x are points where the solution (`var_plot`) is evaluated.

`custom_plot_2D(x, var_plot, solver, levels=None, clims=None)`: a quick 2D plot using the given data `(x, var_plot)`. This function allows arbitrary data to be plotted rapidly. The default Quail visualizers are more detailed and may take longer. If there is no specific mesh to be plotted, one can use `viz` instead to automatically obtain a quick mesh.

`plot_mean(x, q, clims)`: the fastest 2D plot (plots only the mean value of each element). This can be used if `viz` is taking too long, or there's just TOO much detail.

`plot_mean1D(x, q, clims, xscale=1.0, xshift=0.0)`: plots 1D data as 2D rectangles. The xscale and xshift modify the width of the rectangles and where the domain ends. This can be used to make plots containing 2D and 1D portions.

`generate_anim(atm_names, conduit_names, outfilename, num_frames, plot_qty, filter=lambda x, x0:x, initial=1, stride=1, is_high_detail=False)`: generates an animation using ffmpeg. ffmpeg must be installed, and the path must be provided in the environment. List of file names (excluding the number of the file) in the working directory are specified as `atm_names` and `conduit_names`. The output file name for the animation is `outfilename`. Provide also the number of frames to render (`num_frames`), and the name of the quantity (`plot_qty`). Optionally, a filter can be provided (mapping x -> f(x); some useful ones are log10(abs(x))). 

### Exact solution

`util.py/RiemannSolution`: contains fast construction of the solution to the Riemann problem, consisting of an expansion, contact (a.k.a. material interface), and shock.