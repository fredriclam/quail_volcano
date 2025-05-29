// Submarine floor mesh (flat topography with conduit lead-in)
// Large 2D domain. The closest region is intended for use + a buffer region
// if needed. One-domain is useful for parallel EOS evaluation.

// Current setting: 1655 elements @ dx=60, r1=3000

SetFactory("Built-in");
//+

// Frequency mesh size conversion:
//   100 Hz, 1500 m/s: 15 m
//   50 Hz,  1500 m/s: 30 m
//   20 Hz,  1500 m/s: 75 m

dx = 60;

r1 = 3000; // Domain size
a = 50;    // Conduit radius

// Set corner radius
b = 0.5*a;
// Set trig constants for 0 slope
ccos = 1;
ssin = 0;
// Fix initial point on the flank (tangent to flank and vent rounding)
xc = a+b+b*ssin;
yc = b*ccos-b;
// Projection of flank on r = r1: (yc-yp)/(xc-xp) = h, and xp^2 + yp^2 = r1^2
h = 0;
qa = 1+h^2;
qb = 2*h*(yc-h*xc);
qc = (h^2*xc^2-2*h*xc*yc+yc^2-r1^2);
xp = (-qb+Sqrt(qb^2 - 4*qa*qc))/(2*qa);
yp = yc + h*(xp - xc);
// To compute points at r = r_i on the flank:
// xp_i = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r_i^2)))/(2*qa);
// yp_i = yc + h*(xp2 - xc);

// Set upper conduit length
L = 150;

// Convex coefficient for mesh control point placement: closeness to outside
theta1 = 0.6;
theta2 = 0.75;
coarsening_factor = 3.8;

// Define inlet points (r < r1)
Point(1) = {0, r1, 0, coarsening_factor*dx};  // Far top point
Point(2) = {0, 0, 0, dx};
Point(3) = {xp, yp, 0, coarsening_factor*dx};   // Far downflank point
Point(4) = {theta2*xp + (1.0 - theta2)*xc,
            theta2*yp + (1.0 - theta2)*yc, 0, dx}; // Point 1 along flank
Point(5) = {theta1*xp + (1.0 - theta1)*xc,
            theta1*yp + (1.0 - theta1)*yc, 0, dx}; // Point 2 along flank
Point(6) = {xc, yc, 0, dx};
Point(7) = {a+b, -b, 0, dx};
Point(8) = {a, -b, 0, dx};
Point(9) = {a, -L, 0, dx};
Point(10) = {0, -L, 0, dx};
Point(11) = {0, theta1*r1 + (1.0 - theta1)*(-L), 0, dx}; // Point 1 along symmetry line
Point(12) = {0, theta2*r1 + (1.0 - theta2)*(-L), 0, dx}; // Point 2 along symmetry line

// Define inlet curves
Circle(1) = {1, 2, 3};    // First domain boundary r1
Line(2)   = {3, 4};       // Slope (volcano flank)
Line(3)   = {4, 5, 6};    // Slope (volcano flank) -- ignores middle point
Circle(4) = {6, 7, 8};    // Vent corner circle
Line(5)   = {8, 9};       // Right conduit wall
Line(6)   = {9, 10};      // Horizontal 2D1D boundary
Line(7)   = {10, 11, 12}; // Symmetry line (from lowest point to y = r1)  -- ignores middle point
Line(8)   = {12, 1};      // Symmetry line (from lowest point to y = r1)
// Define curve loop from curves
Curve Loop(1) = {4, 5, 6, 7, 8, 1, 2, 3};

// Set planes with compatible (reversed) orientation
Plane Surface(1) = {-1};

// Generate 2D mesh
Mesh 2;
// RefineMesh;

// Mesh export
Physical Curve("r1",       1) = {1};
Physical Curve("ground",   2) = {2};
Physical Curve("ground2",  3) = {3};
Physical Curve("flare",    4) = {4};
Physical Curve("pipewall", 5) = {5};
Physical Curve("x2",       6) = {6};
Physical Curve("symmetry", 7) = {7};
Physical Curve("symmetry2", 8) = {8};
Physical Surface("domain1") = {1};
Save "submarine_one_block.msh";