# Globals for synchronizing input files 
ElementOrder = 1
InitialCondition = {
	"Function" : "IsothermalAtmosphere",
  "h0": -5000, # TODO: fix -4439
  "hmax": 15000,
  "p_atm": 1.7666863813 * 1e5,
  "massFracM": 1e-5,
}
SlipWallQ = {
  "BCType" : "SlipWall",
  "use_stagnation_correction": True,
  "Q": 2.0,
}
# Output file prefix (no trailing underscore)
file_prefix = "refblastA"
write_interval = 100
# Mesh file prefix (no trailing underscore) with parts 1-9
mesh_prefix = "conicalB"