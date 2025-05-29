// Submarine elliptic crater mesh

SetFactory("Built-in");
//+

// Frequency mesh size conversion:
//   100 Hz, 300 m/s: 3 m
//   50 Hz,  300 m/s: 6 m
//   20 Hz,  300 m/s: 15 m
//
//   100 m,  300 m/s: 3 Hz
//   150 m,  300 m/s: 2 Hz
//   200 m,  300 m/s: 1.5 Hz
//   300 m,  300 m/s: 1 Hz
//   375 m,  300 m/s: 0.8 Hz
//   400 m,  300 m/s: 0.75 Hz
//   600 m,  300 m/s: 0.5 Hz
//   750 m,  300 m/s: 0.4 Hz

// Pointwise dx control
dx = 750;      // Pointwise dx control

r1 = 20000; // Domain size
WLH = 300;  // Water layer height
jet_radius = 100;

Point(1) = {0, r1, 0, dx};    // Far top point
Point(2) = {0, 0, 0, dx};     // Boundary circle center
Point(3) = {r1, 0, 0, dx};    // Far right point
Point(4) = {jet_radius, 0, 0, dx};
Point(5) = {0, 0, 0, dx};    // Origin (along symmetry axis)

Circle(1)  = {1, 2, 3};       // First domain boundary r1
Line(2)    = {3, 4};          // Ocean surface
Line(3)   = {4, 5};         // Jet inlet
Line(4)   = {5, 1};         // Symmetry line

// Define curve loop from curves
Curve Loop(1) = {3, 4, 1, 2};

// Set planes with compatible (reversed) orientation
Plane Surface(1) = {-1};

// Set refinement near axis
Field[1] = Distance;
Field[1].CurvesList = {4};
Field[2] = MathEval;
Field[2].F = Sprintf("(F1/100)^2 + %g", 0.25*dx);
// Set refinement near inlet
Field[3] = Distance;
Field[3].CurvesList = {3};
Field[4] = MathEval;
Field[4].F = Sprintf("((F3/2500)^2 + 0.5) * %g", jet_radius);
// Limit mesh size >= dx_L1 in magic box
// Set min field
Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

// Generate 2D mesh
Mesh 2;

// RefineMesh;
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";

// Mesh export
Physical Curve("r1", 1) = {1};
Physical Curve("surface", 2) = {2};
// Physical Curve("flareouter", 3) = {3};
// Physical Curve("x2", 4) = {4};
// Physical Curve("dikeouter", 4) = {4};
// Physical Curve("chamberwallouter", 5) = {5};
// Physical Curve("chambersymmetry", 6) = {6};
// Physical Curve("chamberwallinner", 7) = {7};
// Physical Curve("dikeinner", 8) = {8};
// Physical Curve("flareinner", 9) = {9};
Physical Curve("x2", 10) = {3};
Physical Curve("symmetry", 11) = {4};

Physical Surface("domain1") = {1};

Save "submarine_conduitjet.msh";