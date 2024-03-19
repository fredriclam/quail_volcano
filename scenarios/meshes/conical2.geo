SetFactory("Built-in");
//+

// Set characteristic mesh size with constraint for 20 Hz, 300 m/s: >~ 15 m
// Set characteristic mesh size with constraint for 5 Hz, 300 m/s: >~ 60 m
dx = 70;

// Set distribution exponent
// (alpha = 1 for uniform; alpha = 2 for asymptotically balanced regions--to check)
alpha = 2;
// Set size of interior of domain (excluding buffer zone)
interior_size = 4000;

// Set vent region size
r1 = interior_size*(1-(5/6)^alpha) * 0.4;
// Set conduit radius
a = 50;
// Set corner radius
b = 0.5*a;
// Set trig constants for 1:2 slope
ccos = 2/Sqrt(5);
ssin = 1/Sqrt(5);
// Fix initial point on the flank (tangent to flank and vent rounding)
xc = a+b+b*ssin;
yc = b*ccos-b;
// Projection of flank on r = r1: (yc-yp)/(xc-xp) = h, and xp^2 + yp^2 = r1^2
h = -1/2;
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

// Define inlet points (r < r1)
Point(1) = {0, r1, 0, 0.8*dx};
Point(2) = {0, 0, 0, 0.8*dx};
Point(3) = {xp, yp, 0, 0.8*dx};
Point(4) = {xc, yc, 0, dx/10};
Point(5) = {a+b, -b, 0, 0.8*dx};
Point(6) = {a, -b, 0, dx/10};
Point(7) = {a, -L, 0, dx/5};
Point(8) = {0, -L, 0, dx/5};

// Define outer domain points
// Distributing r ~ n^2 for load balancing
r2 = interior_size*(1-(4/6)^alpha) * 0.55; // * 0.65;
r3 = interior_size*(1-(3/6)^alpha) * 0.70;
r4 = interior_size*(1-(2/6)^alpha) * 0.75;
r5 = interior_size*(1-(1/6)^alpha) * 0.80;
r6 = interior_size * 0.90;
r7 = interior_size;
r8 = 1.2*interior_size;
r9 = 2.5*interior_size;
// Set local mesh size
size2 = 0.8* dx; // 50;
size3 = dx; // 100;
size4 = dx; // 100;...
size5 = dx;
size6 = dx;
// Buffer region
size7 = dx;
size8 = 3*dx;
size9 = 10*dx;

Point(9) = {0, r2, 0, size2};
xp2 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r2^2)))/(2*qa);
yp2 = yc + h*(xp2 - xc);
Point(10) = {xp2, yp2, 0, size2};

Point(11) = {0, r3, 0, size3};
xp3 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r3^2)))/(2*qa);
yp3 = yc + h*(xp3 - xc);
Point(12) = {xp3, yp3, 0, size3};

Point(13) = {0, r4, 0, size4};
xp4 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r4^2)))/(2*qa);
yp4 = yc + h*(xp4 - xc);
Point(14) = {xp4, yp4, 0, size4};

Point(15) = {0, r5, 0, size5};
xp5 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r5^2)))/(2*qa);
yp5 = yc + h*(xp5 - xc);
Point(16) = {xp5, yp5, 0, size5};

Point(17) = {0, r6, 0, size6};
xp6 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r6^2)))/(2*qa);
yp6 = yc + h*(xp6 - xc);
Point(18) = {xp6, yp6, 0, size6};

Point(19) = {0, r7, 0, size7};
xp7 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r7^2)))/(2*qa);
yp7 = yc + h*(xp7 - xc);
Point(20) = {xp7, yp7, 0, size7};

Point(21) = {0, r8, 0, size8};
xp8 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r8^2)))/(2*qa);
yp8 = yc + h*(xp8 - xc);
Point(22) = {xp8, yp8, 0, size8};

Point(23) = {0, r9, 0, size9};
xp9 = (-qb+Sqrt(qb^2 - 4*qa*(h^2*xc^2-2*h*xc*yc+yc^2-r9^2)))/(2*qa);
yp9 = yc + h*(xp9 - xc);
Point(24) = {xp9, yp9, 0, size9};

// Define inlet curves
Circle(1) = {1, 2, 3};
Line(2) = {3, 4};
Circle(3) = {4, 5, 6};
Line(4) = {6, 7};
Line(5) = {7, 8};
Line(6) = {8, 1};
Curve Loop(1) = {7, 9, 2, 3, 4, 5, 6, 8};
Curve Loop(2) = {3, 4, 5, 6, 1, 2};

// Define outer curves
// Region r1-r2
Circle(7) = {9, 2, 10};
Line(8) = {1, 9};
Line(9) = {10, 3};
Circle(10) = {3, 2, 1};
Curve Loop(3) = {8, 7, 9, 10};
// Region r2-r3
Circle(11) = {11, 2, 12};
Line(12) = {12, 10};
Circle(13) = {10, 2, 9};
Line(14) = {9, 11};
Curve Loop(4) = {11, 12, 13, 14};
// Region r3-r4
Circle(15) = {13, 2, 14};
Line(16) = {14, 12};
Circle(17) = {12, 2, 11};
Line(18) = {11, 13};
Curve Loop(5) = {15, 16, 17, 18};
// Region r4-r5
Circle(19) = {15, 2, 16};
Line(20) = {16, 14};
Circle(21) = {14, 2, 13};
Line(22) = {13, 15};
Curve Loop(6) = {19, 20, 21, 22};
// Region r5-r6
Circle(23) = {17, 2, 18};
Line(24) = {18, 16};
Circle(25) = {16, 2, 15};
Line(26) = {15, 17};
Curve Loop(7) = {23, 24, 25, 26};
// Region r6-r7
Circle(27) = {19, 2, 20};
Line(28) = {20, 18};
Circle(29) = {18, 2, 17};
Line(30) = {17, 19};
Curve Loop(8) = {27, 28, 29, 30};
// Region r7-r8
Circle(31) = {21, 2, 22};
Line(32) = {22, 20};
Circle(33) = {20, 2, 19};
Line(34) = {19, 21};
Curve Loop(9) = {31, 32, 33, 34};
// Region r8-r9
Circle(35) = {23, 2, 24};
Line(36) = {24, 22};
Circle(37) = {22, 2, 21};
Line(38) = {21, 23};
Curve Loop(10) = {35, 36, 37, 38};
// Set planes with compatible (reversed) orientation
Plane Surface(2) = {-2};
Plane Surface(3) = {-3};
Plane Surface(4) = {-4};
Plane Surface(5) = {-5};
Plane Surface(6) = {-6};
Plane Surface(7) = {-7};
Plane Surface(8) = {-8};
Plane Surface(9) = {-9};
Plane Surface(10) = {-10};
//+

// Refinement with distance from axis, near conduit
Field[1] = Distance;
Field[1].CurvesList = {6};
Field[2] = MathEval;
Field[2].F = Sprintf("((F1/%g)^2 + 1) * %g", 6*a, 0.15*dx);

// Set refinement near corner radius
Field[3] = Distance;
Field[3].CurvesList = {3};
Field[4] = MathEval;
Field[4].F = Sprintf("((F3/%g)^2 + 1) * %g", 0.5*b, 0.035*dx);

// Refinement with distance from axis, in r1r2 region
Field[5] = Distance;
Field[5].CurvesList = {8};
Field[6] = MathEval;
Field[6].F = Sprintf("((F5/%g)^2 + 1) * %g", 6*a, 0.6*dx);

// Set refinement near axis
Field[7] = Distance;
Field[7].CurvesList = {11};
Field[8] = MathEval;
Field[8].F = Sprintf("(F7/400)^2 + %g", dx);

Field[9] = Box;
Field[9].VIn  = dx/10;    // dx in box
Field[9].VOut = 10*dx;  // dx outside box
Field[9].XMin = 0;
Field[9].XMax = r1;
Field[9].YMin = -L;
Field[9].YMax = 3*L;

// Set min field
Field[10] = Min;
Field[10].FieldsList = {2,4,6};
Background Field = 10;

// Generate 2D mesh
Mesh 2;

// RefineMesh;
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";

// Mesh export

Physical Curve("r1",       1) = {1};
Physical Curve("ground",   2) = {2};
Physical Curve("flare",    3) = {3};
Physical Curve("pipewall", 4) = {4};
Physical Curve("x2",       5) = {5};
Physical Curve("symmetry", 6) = {6};
Physical Surface("domain1") = {2};
Save "conical2_1.msh";
//+
Delete Physicals;
Physical Curve("r2",        7) = {7};
Physical Curve("symmetry2", 8) = {8};
Physical Curve("ground2",   9) = {9};
Physical Curve("r1",        10) = {10};
Physical Surface("domain2") = {3};
Save "conical2_2.msh";
//+
Delete Physicals;
Physical Curve("r3",        11) = {11};
Physical Curve("ground3",   12) = {12};
Physical Curve("r2",        13) = {13};
Physical Curve("symmetry3", 14) = {14};
Physical Surface("domain3") = {4};
Save "conical2_3.msh";
//+
Delete Physicals;
Physical Curve("r4",        15) = {15};
Physical Curve("ground4",   16) = {16};
Physical Curve("r3",        17) = {17};
Physical Curve("symmetry4", 18) = {18};
Physical Surface("domain4") = {5};
Save "conical2_4.msh";
//+
Delete Physicals;
Physical Curve("r5",        19) = {19};
Physical Curve("ground5",   20) = {20};
Physical Curve("r4",        21) = {21};
Physical Curve("symmetry5", 22) = {22};
Physical Surface("domain5") = {6};
Save "conical2_5.msh";
//+
Delete Physicals;
Physical Curve("r6",        23) = {23};
Physical Curve("ground6",   24) = {24};
Physical Curve("r5",        25) = {25};
Physical Curve("symmetry6", 26) = {26};
Physical Surface("domain6") = {7};
Save "conical2_6.msh";
//+
Delete Physicals;
Physical Curve("r7",        27) = {27};
Physical Curve("ground7",   28) = {28};
Physical Curve("r6",        29) = {29};
Physical Curve("symmetry7", 30) = {30};
Physical Surface("domain7") = {8};
Save "conical2_7.msh";
//+
Delete Physicals;
Physical Curve("r8",        31) = {31};
Physical Curve("ground8",   32) = {32};
Physical Curve("r7",        33) = {33};
Physical Curve("symmetry8", 34) = {34};
Physical Surface("domain8") = {9};
Save "conical2_8.msh";
//+
Delete Physicals;
Physical Curve("r9",        35) = {35};
Physical Curve("ground9",   36) = {36};
Physical Curve("r8",        37) = {37};
Physical Curve("symmetry9", 38) = {38};
Physical Surface("domain9") = {10};
Save "conical2_9.msh";