SetFactory("Built-in");
//+
// Inlet points
Point(1) = {0, 150, 0, 10};
Point(2) = {0, 0, 0, 50};
Point(3) = {150, 0, 0, 10};
Point(4) = {100, 0, 0, 10};
Point(5) = {100, -50, 0, 50};
Point(6) = {50, -50, 0, 10};
Point(7) = {50, -150, 0, 20};
Point(8) = {0, -150, 0, 20};
//+
// Outer domain points
Point(9) = {0, 300, 0, 20};
Point(10) = {300, 0, 0, 20};
Point(11) = {0, 900, 0, 100};
Point(12) = {900, 0, 0, 100};
Point(13) = {0, 2000, 0, 120};
Point(14) = {2000, 0, 0, 120};
Point(15) = {0, 7500, 0, 1000};
Point(16) = {7500, 0, 0, 1000};
Point(17) = {0, 30000, 0, 3000};
Point(18) = {30000, 0, 0, 3000};
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
Save "volcanoA1.msh";
//+
Delete Physicals;
Physical Curve("r1", 17) = {10};
Physical Curve("r2", 14) = {7};
Physical Curve("symmetry_far", 15) = {8};
Physical Curve("ground_far", 16) = {9};
Physical Surface("domain2", 3) = {3};
Save "volcanoA2.msh";
//+
Delete Physicals;
Physical Curve("r3", 18) = {11};
Physical Curve("ground3", 19) = {12};
Physical Curve("r2", 20) = {13};
Physical Curve("symmetry3", 21) = {14};
Physical Surface("domain3", 4) = {4};
Save "volcanoA3.msh";
//+
Delete Physicals;
Physical Curve("r4", 22) = {15};
Physical Curve("ground4", 23) = {16};
Physical Curve("r3", 24) = {17};
Physical Curve("symmetry4", 25) = {18};
Physical Surface("domain4", 5) = {5};
Save "volcanoA4.msh";
//+
Delete Physicals;
Physical Curve("r5", 26) = {19};
Physical Curve("ground5", 27) = {20};
Physical Curve("r4", 28) = {21};
Physical Curve("symmetry5", 29) = {22};
Physical Surface("domain5", 6) = {6};
Save "volcanoA5.msh";
//+
Delete Physicals;
Physical Curve("r6", 30) = {23};
Physical Curve("ground6", 31) = {24};
Physical Curve("r5", 32) = {25};
Physical Curve("symmetry6", 33) = {26};
Physical Surface("domain6", 7) = {7};
Save "volcanoA6.msh";//+
Hide "*";
//+
Show {
  Point{1}; Point{3}; Point{4}; Point{6}; Point{7}; Point{8}; Curve{1}; Curve{2}; Curve{3}; Curve{4}; Curve{5}; Curve{6}; Surface{2}; 
}
