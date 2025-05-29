// Submarine elliptic crater mesh

SetFactory("Built-in");
//+

// Frequency mesh size conversion:
//   100 Hz, 1500 m/s: 15 m
//   50 Hz,  1500 m/s: 30 m
//   20 Hz,  1500 m/s: 75 m
//
//   100 m,  1500 m/s: 15 Hz
//   150 m,  1500 m/s: 10 Hz
//   200 m,  1500 m/s: 7.5 Hz
//   300 m,  1500 m/s: 5 Hz
//   375 m,  1500 m/s: 4 Hz
//   400 m,  1500 m/s: 3.75 Hz
//   750 m,  1500 m/s: 2 Hz
//   800 m,  1500 m/s: 1.875 Hz
//   1500 m,  1500 m/s: 1 Hz

// Pointwise dx control
dx = 750;      // Pointwise dx control
// Box dx control
dx_L1 = 100; // Refinement x2 // 75;     // In-box (region of interest) dx control -- chamber, dike
dx_L2 = 200; // Refinement ~ // 150;   // Transition level 1

r1 = 20000; // Domain size
WLH = 300;  // Water layer height
jet_width = 50; // Jet width

a = 2000;      // Chamber horizontal semi axis
b = 800;       // Chamber vertical semi axis
c1 = 0.6*a;    // Dike inner x-point (line projects to this number on x-axis)
c2 = c1 + jet_width;  // Dike outer x-point (line projects to this number on x-axis)
H = 2500;      // Depth of center of ellipsoial chamber
R = 50;        // Corner fillet radius generic
R1 = R;        // Corner fillet radius inner
R2 = R;        // Corner fillet radius outer
theta = 89.99*Pi/180; // Dip angle (rad)
s = Tan(theta);       // Slope of dip

// Compute plumbing system coordinates relative to center of chamber //

// Quadratic formula temps
_a1 = a*a/(b*b)*s*s + 1;
_b1 = -2*c1*a*a/(b*b)*s*s;
_c1 = a*a/(b*b)*c1*c1*s*s - a*a;
// Compute intersection of ellipsoidal chamber with dike inner line
inner_dike_x = (-_b1 + Sqrt(_b1*_b1 - 4*_a1*_c1)) / (2*_a1);
inner_dike_y = (inner_dike_x - c1) * s;
// Quadratic formula temps
_a2 = a*a/(b*b)*s*s + 1;
_b2 = -2*c2*a*a/(b*b)*s*s;
_c2 = a*a/(b*b)*c2*c2*s*s - a*a;
// Compute intersection of ellipsoidal chamber with dike outer line
outer_dike_x = (-_b2 + Sqrt(_b2*_b2 - 4*_a2*_c2)) / (2*_a2);
outer_dike_y = (outer_dike_x - c2) * s;
// Compute fillet circle for tangent on lower hemicircle
inner_dike_fillet_y = H - R1 * (1 - Cos(Pi - theta));
inner_dike_fillet_x = c1 + inner_dike_fillet_y / s;
inner_dike_fillet_center_y = H - R1;
inner_dike_fillet_center_x = inner_dike_fillet_x - R1 * Sin(Pi - theta);
// Compute fillet circle for tangent on upper hemicircle
outer_dike_fillet_y = H - R2 * (1 - Cos(theta));
outer_dike_fillet_x = c2 + outer_dike_fillet_y / s;
outer_dike_fillet_center_y = H - R2;
outer_dike_fillet_center_x = outer_dike_fillet_x + R2 * Sin(theta);


Point(1) = {0, r1, 0, dx};    // Far top point
Point(2) = {0, 0, 0, dx};     // Boundary circle center
Point(3) = {r1, 0, 0, dx};    // Far right point
Point(4) = {outer_dike_fillet_center_x, 0, 0, dx};               // Outer fillet x floor
Point(5) = {outer_dike_fillet_center_x, -R, 0, dx};              // Center for corner rounding
Point(6) = {outer_dike_fillet_x, outer_dike_fillet_y-H, 0, dx};  // End of corner rounding
// Point(7) = {outer_dike_x, outer_dike_y-H, 0, dx};                // Dike outer x chamber
// Point(8) = {0, -H, 0, dx};    // Ellipse center
// Point(9) = {a, -H, 0, dx};    // Ellipse major axis point
// Point(10) = {0, -H-b, 0, dx}; // Ellipse bottom
// Point(11) = {0, -H+b, 0, dx}; // Ellipse top
// Point(12) = {inner_dike_x, inner_dike_y-H, 0, dx};               // Dike inner x chamber
Point(13) = {inner_dike_fillet_x, inner_dike_fillet_y-H, 0, dx}; // End of corner rounding
Point(14) = {inner_dike_fillet_center_x, -R, 0, dx};             // Center for corner rounding
Point(15) = {inner_dike_fillet_center_x, 0, 0, dx};              // Inner fillet x floor
Point(16) = {0, 0, 0, dx};    // Origin (along symmetry axis)
x_mean = 0.5*inner_dike_fillet_center_x + 0.5 * outer_dike_fillet_center_x;
Point(17) = {x_mean, 0.0, 0, dx}; // Refinement point 1
Point(18) = {x_mean, 0.7*x_mean, 0, dx}; // Refinement point 2

Circle(1)  = {1, 2, 3};        // First domain boundary r1
Line(2)    = {3, 4};           // Ocean floor outer
Ellipse(3) = {4, 5, 6, 6};     // Dike outer fillet
// Line(4)    = {6, 7};           // Dike outer
// Ellipse(5) = {7, 8, 9, 10};    // Outer ellipse part
// Line(6)    = {10, 11};         // Ellipse x symmetry axis
// Ellipse(7) = {11, 8, 9, 12};   // Inner ellipse part
// Line(8)    = {12, 13};         // Dike inner
Line(4) = {6, 13};                // Inlet coupling boundary
Ellipse(9) = {13, 14, 15, 15}; // Dike inner fillet
Line(10)   = {15, 16};         // Ocean floor inner
Line(11)   = {16, 1};          // Symmetry line
Line(12)   = {17, 18}; // Refinement line

// Define curve loop from curves
Curve Loop(1) = {3, 4, 9, 10, 11, 1, 2};

// Set planes with compatible (reversed) orientation
Plane Surface(1) = {-1};

// Set refinement from refinement curve
Field[1] = Distance;
Field[1].CurvesList = {12};
Field[2] = MathEval;
Field[2].F = Sprintf("(F1/100)^2 + %g", 0.5*dx_L2);
// Set refinment near inlet
Field[3] = Distance;
Field[3].CurvesList = {4};
Field[4] = MathEval;
Field[4].F = Sprintf("(F3/100)^2 + %g", 0.3*dx_L1);
// Limit mesh size >= dx_L1 in magic box
Field[6] = Box;
Field[6].VIn  = dx_L1;
Field[6].VOut = 0.001;
Field[6].XMin = 0;
Field[6].XMax = 1.5*c2;
Field[6].YMin = -R;
Field[6].YMax = 4*c2;
// Set max
Field[7] = Max;
Field[7].FieldsList={2, 6};

// Set min field
Field[8] = Min;
Field[8].FieldsList = {4,7};
Background Field = 8;

// Generate 2D mesh
Mesh 2;

// RefineMesh;
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";

// Mesh export
Physical Curve("r1", 1) = {1};
Physical Curve("groundouter", 2) = {2};
Physical Curve("flareouter", 3) = {3};
Physical Curve("x2", 4) = {4};
// Physical Curve("dikeouter", 4) = {4};
// Physical Curve("chamberwallouter", 5) = {5};
// Physical Curve("chambersymmetry", 6) = {6};
// Physical Curve("chamberwallinner", 7) = {7};
// Physical Curve("dikeinner", 8) = {8};
Physical Curve("flareinner", 9) = {9};
Physical Curve("groundinner", 10) = {10};
Physical Curve("symmetry", 11) = {11};

Physical Surface("domain1") = {1};

Save "submarine_ringfault_above_ocean.msh";