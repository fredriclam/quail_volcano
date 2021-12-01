// Gmsh project created on Fri Jul 30 21:33:03 2021
//+
Point(1) = {5, 0, 0, 1.0};
//+
Point(2) = {5, 5, 0, 1.0};
//+
Point(3) = {-5, 5, 0, 1.0};
//+
Point(4) = {-5, 0, 0, 1.0};
//+
Point(5) = {-1, 0, 0, 1.0};
//+
Point(6) = {1, 0, 0, 1.0};
//+
Point(7) = {1, -3, 0, 1.0};
//+
Point(8) = {-1, -3, 0, 1.0};
//+
Line(1) = {3, 4};
//+
Line(2) = {4, 5};
//+
Line(3) = {5, 8};
//+
Line(4) = {8, 7};
//+
Line(5) = {7, 6};
//+
Line(6) = {6, 1};
//+
Line(7) = {1, 2};
//+
Line(8) = {2, 3};
//+
Curve Loop(1) = {2, 3, 4, 5, 6, 7, 8, 1};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, 1} {
  Surface{1}; 
  Layers{1};
  Recombine;
}
//+
Physical Surface("right", 51) = {41};
//+
Physical Surface("top", 52) = {45};
//+
Physical Surface("left", 53) = {49};
//+
Physical Surface("botwallleft", 54) = {21};
//+
Physical Surface("botwallright", 55) = {37};
//+
Physical Surface("entryleft", 56) = {25};
//+
Physical Surface("entryright", 57) = {33};
//+
Physical Surface("entrybot", 58) = {29};
//+
Physical Surface("front", 59) = {50};
//+
Physical Surface("back", 60) = {1};
//+
Physical Volume("volume", 61) = {1};
