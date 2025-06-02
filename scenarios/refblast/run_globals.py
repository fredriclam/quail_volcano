# Globals for synchronizing input files 
ElementOrder = 1
InitialCondition = {
	"Function" : "IsothermalAtmosphere",
  "h0": -150,
  "hmin": -10000,
  "hmax": 15000,
  "p_atm": 1.01325e5,
  "massFracWv": 1e-5,
  "massFracM": 1e-5,
  "T": 298.15,
}
SlipWallQ = {
  "BCType" : "SlipWall",
  "use_stagnation_correction": False,
}
# Output file prefix (no trailing underscore)
file_prefix = "refblastH3" #H2: check 1D conduit
write_interval = 100
# Mesh file prefix (no trailing underscore) with parts 1-9
mesh_prefix = "conicalB"
