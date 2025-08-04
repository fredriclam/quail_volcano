// Gmsh .geo file for a vertical 1D conduit mesh with triangular elements
SetFactory("Built-in");

// Set characteristic mesh size
dx = 1;

// Set conduit and domain parameters
a = 10;  // Domain width
L = 0;  // Bottom coordinate
H = 205; // Top coordinate

// Define points (merge conduit and domain points)
Point(1) = {0, -L, 0, dx}; // Bottom-left (conduit bottom and domain corner)
Point(2) = {0, H, 0, dx};  // Top-left (conduit top and domain corner)
Point(3) = {a, -L, 0, dx}; // Bottom-right
Point(4) = {a, H, 0, dx};  // Top-right

// Define conduit curve
//Line(1) = {1, 2}; // Vertical conduit

// Define rectangular domain boundary curves
Line(1) = {1, 3}; // Bottom
Line(2) = {3, 4}; // Right
Line(3) = {4, 2}; // Top
Line(4) = {2, 1}; // Left

// Define curve loop for the rectangular domain
Curve Loop(1) = {1, 2, 3, 4};

Plane Surface(1) = {1};

// Set physical groups
Physical Curve("bottom", 1) = {1};     // Bottom boundary
Physical Curve("right", 2) = {2};      // Right boundary
Physical Curve("top", 3) = {3};        // Top boundary
Physical Curve("symmetry", 4) = {4};       // Left boundary
Physical Surface("domain1") = {1};  // Domain surface

// Mesh settings
Mesh.Algorithm = 6; // Frontal-Delaunay for better quality
Mesh.ElementOrder = 1; // Ensure linear elements if required
Mesh.Optimize = 1; // Optimize mesh quality
Mesh 2; // Generate 2D mesh

// Save mesh
Save "vertical_conduit.msh";