SetFactory("Built-in");
//+
// Inlet points
Point(1) = {0, 1, 0, 0.05};
Point(2) = {1, 1, 0, 0.05};
Point(3) = {1, 0, 0, 0.05};
Point(4) = {0, 0, 0, 0.05};
//+
// Rectangle boundary
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
//+
// Set planes with compatible (reversed) orientation
Plane Surface(1) = {-1};
//Plane Surface(1) = {1};
Mesh 2;
RefineMesh;
//+
Physical Curve("y2", 1) = {1};
Physical Curve("x2", 2) = {2};
Physical Curve("y1", 3) = {3};
Physical Curve("x1", 4) = {4};
Physical Surface("domain1") = {1};
Save "rectangle.msh";
//+