import tkinter
from tkinter import font
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Local imports
import atomics

def compute_state(data):
    # Cast data
    def float_else_zero(x):
        try:
            return float(x)
        except ValueError:
            return 1e-10
    data_clean = [float_else_zero(x) for x in data]
    # Unpack data vector using manual ordering
    p, T, yA, yWv = data_clean
    # Manual physics parameters
    class Physics_():
        def __init__(self):
            self.Gas=[{"R": 287., "gamma": 1.4, "c_v": 287. / (1.4 - 1)},
                {"R": 8.314/18.02e-3, "c_p": 2.288e3, "c_v": 2.288e3 - 8.314/18.02e-3}]
            self.Liquid={"K": 10e9, "rho0": 2.5e3, "p0": 5e6,
                            "E_m0": 0, "c_m": 3e3}
            self.Solubility={"k": 5e-6, "n": 0.5}
    physics = Physics_()
    # Phasic densities
    rhoA = p / (physics.Gas[0]["R"] * T)
    rhoWv = p / (physics.Gas[1]["R"] * T)
    rhoM = physics.Liquid["rho0"] * (
        1.0 + (p - physics.Liquid["p0"]) / physics.Liquid["K"])
    rhoVec = np.array([rhoA, rhoWv, rhoM])
    # Phasic specific volume summation
    yM = 1.0 - yA - yWv
    v = yA / rhoA + yWv / rhoWv + yM / rhoM
    rho = 1.0 / v
    # Vector of partial densities
    arhoVec = np.array([yA, yWv, yM]) * rho
    # Internal energy density of mixture
    e = atomics.c_v(arhoVec, physics)[0] * T
    # Gas volume fraction
    phi = atomics.gas_volfrac(arhoVec, T, physics)[0]
    # Dissolved water partial density for saturation
    sat = physics.Solubility["k"] * p ** physics.Solubility["n"]
    arhoWd = sat / (1.0 + sat) * arhoVec[2]
    arhoWt = arhoWd + arhoVec[1]

    Gamma = atomics.Gamma(arhoVec, physics)[0]
    return (*arhoVec, e, arhoWd, arhoWt), (phi, Gamma)

matplotlib.use('TkAgg')

# Define accessible data
entryboxes = []
plot_handles = []
canvases = []
figs = []
displabels = []
dispstrvars = []
# Define callbacks
def refresh():
    # Get input data
    data = [e.get() for e in entryboxes]
    p0, T0 = data[0:2]
    # Set output display
    output_states, extra_output_states = compute_state(data)
    for i, state in enumerate(output_states):
        dispstrvars[i].set(f"{state:.2f}")
    # Compute variation data  TODO: async compute and draw
    p_range = np.logspace(5.0, 8.0, 300)
    T_range = np.linspace(300, 1000, 51)
    p_mg, T_mg = np.meshgrid(p_range, T_range)
    output = np.zeros((len(output_states)+len(extra_output_states), *p_mg.shape))
    for i, p in enumerate(p_range):
        for j, T in enumerate(T_range):
            output_states, extra_output_states = compute_state(
                [p, T, *data[2:]])
            output[:,j,i] = [*output_states, *extra_output_states[:-1]]
    # Compute isentropic line
    
    # Redraw plots
    for fig in figs:
        fig.clf()
        subplot = fig.add_subplot(111)
        cf = subplot.contourf(p_mg, T_mg, output[-1,:,:])
        subplot.set_xlabel("p (Pa)")
        # subplot.set_xscale("log")
        subplot.set_ylabel("T (K)")
        # subplot.set_ylabel("Mass fraction water, exsolved")
        fig.colorbar(cf, ax=subplot, label="Gas volume fraction")
        # Plot input state
        # subplot.plot(p0, T0, 'r.')
        subplot.scatter([float(p0)], [float(T0)], color="red")

    for canvas in canvases:
        canvas.draw()    

''' tkinter setup '''
# Set tkinter frame
root = tkinter.Tk()
root.geometry("1080x720")
root.config(bg = 'black')
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1, minsize=140)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.title("State converter")

''' Set fonts and styles '''
tkinter.font.nametofont("TkDefaultFont").configure(family="Courier", size=13)

''' Set figure canvas '''
# Set label in which canvas is embedded
drawframe = tkinter.Frame(root, bg="#52a16b")
# Create plt figure
fig = matplotlib.figure.Figure(dpi=144)
subplot = fig.add_subplot(111)
# Plot test data
# subplot.plot([1,], [1,])
canvas = FigureCanvasTkAgg(fig, master=drawframe)
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
# Save handle to plot
figs.append(fig)
plot_handles.append(subplot)
canvases.append(canvas)

''' Create user input widgets '''
entryframecolors = {
    "primary": "#52a16b",
    "secondary": "#9ae3b1",
    "tertiary": "#edc07e",
}
entryframe = tkinter.Frame(root, bg=entryframecolors["primary"])
# Entry boxes
boxlabelnames = ["pressure (Pa)", "temperature (K)", "y, air", "y, water, exsolv"]
defaultvalues = [1e5, 300, 0.0, 0.005]
for i in range(len(boxlabelnames)):
  label_ = tkinter.Label(entryframe,
      textvariable=tkinter.StringVar(value=boxlabelnames[i]),
      bg=entryframecolors["primary"])
  label_.pack(anchor="w", padx=5, pady=5)
  entry_ = tkinter.Entry(entryframe, width=32,
      textvariable=tkinter.StringVar(value=defaultvalues[i]),
      bg=entryframecolors["secondary"])
  entry_.pack(padx=5, pady=5)
  entryboxes.append(entry_)
# Manual refresh button
refreshbutton = tkinter.Button(entryframe, text="Reload \U0001F504",
  fg="black", bg=entryframecolors["tertiary"], command=refresh)
refreshbutton.pack(side = tkinter.RIGHT)

''' Set display (output) widget '''
dispframecolors = {
    "primary": "#52a16b",
    "secondary": "#9ae3b1",
    "tertiary": "#edc07e",
}
dispframe = tkinter.Frame(root, bg=dispframecolors["primary"])
# Entry boxes
displabelnames = ["arhoA", "arhoWv", "arhoM", "e", "arhoWd @ sat", "arhoWt @ sat"]
for i in range(len(displabelnames)):
  label_ = tkinter.Label(dispframe,
      textvariable=tkinter.StringVar(value=displabelnames[i]),
      bg=entryframecolors["primary"])
  label_.pack(anchor="w", padx=5, pady=5)
  strvar_ = tkinter.StringVar(value="0.0")
  label_ = tkinter.Label(dispframe,
      textvariable=strvar_,
      bg=entryframecolors["primary"],
      width=24)
  label_.pack(anchor="c", padx=5, pady=5)
  displabels.append(entry_)
  dispstrvars.append(strvar_)

''' Set widget layout '''
entryframe.grid(column=1, row=0, padx=20)
dispframe.grid(column=1, row=1, padx=20)
drawframe.grid(column=0, row=0, rowspan=2)

''' First refresh '''
refresh()

''' Start interactive loop '''
root.mainloop()