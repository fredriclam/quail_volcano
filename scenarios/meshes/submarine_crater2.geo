// Submarine elliptic crater mesh

SetFactory("Built-in");
//+

// Frequency mesh size conversion:
//   100 Hz, 1500 m/s: 15 m
//   50 Hz,  1500 m/s: 30 m
//   20 Hz,  1500 m/s: 75 m

// Pointwise dx control
dx = 180;      // Pointwise dx control
// Box dx control
dx_in = 60;    // In-box (region of interest) dx control
dx_out = 600;  // Out-box (region of interest) dx control

r1 = 3000; // Domain size

a1 = 500;              // Crater radius (~2 to 4 km in HTHH for example)
crater_depth = 100;     // Crater depth
a2 = 50;                // Corner rounding horizontal axis
b2 = 20;                // Corner rounding vertical axis
b1 = crater_depth - b2; // Account for rounding height in total crater depth

// Fix initial point on the flank (tangent to flank and vent rounding)
xc = a1+a2;
yc = 0;

// Convex coefficient for mesh control point placement: closeness to outside
theta1 = 0.6;
theta2 = 0.75;
coarsening_factor = 1.0;
// coarsening_factor = 3.8;

// Define inlet points (r < r1)
Point(1) = {0, r1, 0, coarsening_factor*dx};   // Far top point
Point(2) = {0, 0, 0, dx};
Point(3) = {r1, 0, 0, coarsening_factor*dx};   // Far downflank point
Point(4) = {theta2*r1 + (1.0 - theta2)*xc,
            0, 0, dx};                         // Point 1 along flank
Point(5) = {theta1*r1 + (1.0 - theta1)*xc,
            0, 0, dx};                         // Point 2 along flank
Point(6) = {a1+a2, 0, 0, dx};                  // Far upflank point
Point(7) = {a1+a2, -b2, 0, dx};                // Center for corner rounding
Point(8) = {a1, -b2, 0, dx};                   // End of corner rounding
// Point(9) = {a, -L, 0, dx};                   // Unused point
// Point(10) = {0, -L, 0, dx};                  // Unused point
Point(11) = {0, -b2, 0, dx};                  // Ellipse center
// Point(12) = {ychamber_majoraxis, ychamber_center, 0, dx}; // Unused point
Point(13) = {0, -b1-b2, 0, dx}; // Ellipse end point
Point(14) = {0, theta1*r1 + (1.0 - theta1)*(-b1-b2), 0, dx}; // Point 1 along symmetry line
Point(15) = {0, theta2*r1 + (1.0 - theta2)*(-b1-b2), 0, dx}; // Point 2 along symmetry line

// Define inlet curves
Circle(1) = {1, 2, 3};    // First domain boundary r1
Line(2)   = {3, 4};       // Slope (volcano flank)
Line(3)   = {4, 5, 6};    // Slope (volcano flank) -- ignores middle point
Ellipse(4) = {6, 7, 8, 8};    // Vent corner circle
//Line(5)   = {8, 9};       // Right conduit wall
// Line(6)   = {9, 10};      // Horizontal 2D1D boundary
Ellipse(6) = {8, 11, 13, 13}; // Crater main ellipse
Line(7)   = {13, 14, 15}; // Symmetry line (from lowest point to y = r1)  -- ignores middle point
Line(8)   = {15, 1};      // Symmetry line (from lowest point to y = r1)
// Define curve loop from curves
Curve Loop(1) = {4, 6, 7, 8, 1, 2, 3};

// Set planes with compatible (reversed) orientation
Plane Surface(1) = {-1};

// Set mesh boxing
Field[1] = Box;
Field[1].VIn = dx_in;    // dx in box
Field[1].VOut = dx_out;  // dx outside box
Field[1].XMin = 0;
Field[1].XMax = r1;
Field[1].YMin = -b1-b2;
Field[1].YMax = 300;
Background Field = 1;

// Generate 2D mesh
Mesh 2;
RefineMesh;

// Mesh export
Physical Curve("r1",       1) = {1};
Physical Curve("ground",   2) = {2};
Physical Curve("ground2",  3) = {3};
Physical Curve("flare",    4) = {4};
//Physical Curve("pipewall", 5) = {5};
// Physical Curve("x2",       6) = {6};
Physical Curve("chamberwall", 6) = {6};
Physical Curve("symmetry", 7) = {7};
Physical Curve("symmetry2", 8) = {8};
Physical Surface("domain1") = {1};
Save "submarine_crater2.msh";