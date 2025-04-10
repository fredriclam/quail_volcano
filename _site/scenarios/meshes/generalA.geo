SetFactory("Built-in");
//+
// Inlet points
Point(1) = {0, 3, 0, 0.2};
Point(2) = {0, 0, 0, 1};
Point(3) = {3, 0, 0, 0.2};
Point(4) = {2, 0, 0, 0.2};
Point(5) = {2, -1, 0, 1};
Point(6) = {1, -1, 0, 0.2};
Point(7) = {1, -3, 0, 0.4};
Point(8) = {0, -3, 0, 0.4};
//+
// Outer domain points
Point(9) = {0, 6, 0, 0.4};
Point(10) = {6, 0, 0, 0.4};
Point(11) = {0, 18, 0, 2.0};
Point(12) = {18, 0, 0, 2.0};
Point(13) = {0, 40, 0, 2.4};
Point(14) = {40, 0, 0, 2.4};
Point(15) = {0, 150, 0, 20.0};
Point(16) = {150, 0, 0, 20.0};
Point(17) = {0, 600, 0, 60.0};
Point(18) = {600, 0, 0, 60.0};
//+
// Inlet curves
Circle(1) = {1, 2, 3};
Line(2) = {3, 4};
Circle(3) = {4, 5, 6};
Line(4) = {6, 7};
Line(5) = {7, 8};
Line(6) = {8, 1};
Curve Loop(1) = {7, 9, 2, 3, 4, 5, 6, 8};
Curve Loop(2) = {3, 4, 5, 6, 1, 2};
//+
// Outer curves, r1-r2
Circle(7) = {9, 2, 10};
Line(8) = {1, 9};
Line(9) = {10, 3};
Circle(10) = {3, 2, 1};
Curve Loop(3) = {8, 7, 9, 10};
//+
// Outer curves, r2-r3
Circle(11) = {11, 2, 12};
Line(12) = {12, 10};
Circle(13) = {10, 2, 9};
Line(14) = {9, 11};
Curve Loop(4) = {11, 12, 13, 14};
//+
// Outer curves, r3-r4
Circle(15) = {13, 2, 14};
Line(16) = {14, 12};
Circle(17) = {12, 2, 11};
Line(18) = {11, 13};
Curve Loop(5) = {15, 16, 17, 18};
//+
// Outer curves, r4-r5
Circle(19) = {15, 2, 16};
Line(20) = {16, 14};
Circle(21) = {14, 2, 13};
Line(22) = {13, 15};
Curve Loop(6) = {19, 20, 21, 22};
// Outer curves, r5-r6
Circle(23) = {17, 2, 18};
Line(24) = {18, 16};
Circle(25) = {16, 2, 15};
Line(26) = {15, 17};
Curve Loop(7) = {23, 24, 25, 26};
//+
// (Disabled) global plane surface
// Plane Surface(1) = {1};
//+
// Embedded line no longer needed; mesh on two separate surfaces
// Curve{1} In Surface{1};
//+
// Set planes with compatible (reversed) orientation
Plane Surface(2) = {-2};
Plane Surface(3) = {-3};
Plane Surface(4) = {-4};
Plane Surface(5) = {-5};
Plane Surface(6) = {-6};
Plane Surface(7) = {-7};
//+
// Set mesh size constraint embeddings
// Line{15} In Surface{2};
// Line{16} In Surface{2};
// Line{17} In Surface{2};
// Line{18} In Surface{2};
//+
Mesh 2;
// RefineMesh;
//+
Physical Curve("r1", 8) = {1};
Physical Curve("ground", 9) = {2};
Physical Curve("flare", 10) = {3};
Physical Curve("pipewall", 11) = {4};
Physical Curve("x2", 12) = {5};
Physical Curve("symmetry", 13) = {6};
Physical Surface("domain1", 2) = {2};
Save "generalA1.msh";
//+
Delete Physicals;
Physical Curve("r1", 17) = {10};
Physical Curve("r2", 14) = {7};
Physical Curve("symmetry_far", 15) = {8};
Physical Curve("ground_far", 16) = {9};
Physical Surface("domain2", 3) = {3};
Save "generalA2.msh";
//+
Delete Physicals;
Physical Curve("r3", 18) = {11};
Physical Curve("ground3", 19) = {12};
Physical Curve("r2", 20) = {13};
Physical Curve("symmetry3", 21) = {14};
Physical Surface("domain3", 4) = {4};
Save "generalA3.msh";
//+
Delete Physicals;
Physical Curve("r4", 22) = {15};
Physical Curve("ground4", 23) = {16};
Physical Curve("r3", 24) = {17};
Physical Curve("symmetry4", 25) = {18};
Physical Surface("domain4", 5) = {5};
Save "generalA4.msh";
//+
Delete Physicals;
Physical Curve("r5", 26) = {19};
Physical Curve("ground5", 27) = {20};
Physical Curve("r4", 28) = {21};
Physical Curve("symmetry5", 29) = {22};
Physical Surface("domain5", 6) = {6};
Save "generalA5.msh";
//+
Delete Physicals;
Physical Curve("r6", 30) = {23};
Physical Curve("ground6", 31) = {24};
Physical Curve("r5", 32) = {25};
Physical Curve("symmetry6", 33) = {26};
Physical Surface("domain6", 7) = {7};
Save "generalA6.msh";//+
Hide "*";
//+
Show {
  Point{1}; Point{3}; Point{4}; Point{6}; Point{7}; Point{8}; Curve{1}; Curve{2}; Curve{3}; Curve{4}; Curve{5}; Curve{6}; Surface{2}; 
}
