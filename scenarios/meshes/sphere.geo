SetFactory("Built-in");
//+

// Set characteristic mesh size with constraint for 20 Hz, 300 m/s: >~ 15 m
dx_max = 100; // Background dx
dx = 5;      // Minimum dx
scale = 100;  // Scale for refinement (meters)

// Set interior sphere size
r1 = 30;
// Set exterior sphere size
r2 = 3000;

// Define inlet points (r < r1)
Point(1) = {1e-10, r1, 0, dx_max};
Point(2) = {0, 0, 0, dx_max};
Point(3) = {1e-10, -r1, 0, dx_max};
Point(4) = {0, -r2, 0, dx_max};
Point(5) = {0, 0, 0, dx_max};
Point(6) = {0, r2, 0, dx_max};

Circle(1) = {1, 2, 3};
Line(2) = {3, 4};
Circle(3) = {4, 5, 6};
Line(4) = {6, 1};

// Define outer curves
Curve Loop(1) = {1, 2, 3, 4};

Plane Surface(1) = {1};

// Set refinement formula from inner radius
Field[1] = Distance;
Field[1].CurvesList = {1};
Field[2] = MathEval;
Field[2].F = Sprintf("(1 + F1/%g) * %g", scale, dx);
Background Field = 2;

// Generate 2D mesh
Mesh 2;

// Mesh export
Physical Curve("r1", 1) = {1};
Physical Curve("symmetrylower",   2) = {2};
Physical Curve("r2", 3) = {3};
Physical Curve("symmetryupper", 4) = {4};
Physical Surface("domain1") = {1};
Save "sphere.msh";