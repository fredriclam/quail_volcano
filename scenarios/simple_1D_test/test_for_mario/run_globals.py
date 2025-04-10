# Global parameters that are relevant to multiple input files 

ElementOrder1D = 0
ElementOrder2D = 0
InitialCondition = {
	"Function" : "IsothermalAtmosphere",
  "h0": -5000,
  "hmax": 15000,
  "p_atm": 1.7666863813 * 1e5,
  "massFracM": 1e-5,
}

# Output file prefix (no trailing underscore)
output_file_prefix = "test_infrasound"
write_interval_2D = 50

# Mesh file prefix (no trailing underscore)
# There are some meshes in the scenarios/meshes/ folder
# tungurahuaA1 to tungurahuaA8 are different parts of the same mesh
# A1 is the section closest to the vent, A8 is the section farthest
mesh_prefix = "tungurahuaA"