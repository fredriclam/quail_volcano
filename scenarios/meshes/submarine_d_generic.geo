// Blasted crater wide vent geometry

SetFactory("Built-in");
//+

// Pointwise dx control (global)
dx = 400;
// Box dx control
dx_L1 = 25;  // Range-based refinement near vent
dx_L2 = 25;  // Box refinement for jet under ocean surface

// Domain size
r1 = 5000;

// Problem specs
conduit_radius = 25;
depth_2D = 100; // Water depth in 2D portion
depth_1D = 150; // Water depth in 1D portion
angle_deg = 15;

// Auto name of target .msh file , e.g., "submarine_atmosphere10_large.msh";
msh_target_name = Sprintf("sub2_atm_%g_%g_%g.msh", conduit_radius, depth_1D, angle_deg);
// Tan of jet angle
tan_angle = Tan(angle_deg * Pi / 180);
// Radius at bottem of the jet opening
interface_radius = conduit_radius + depth_1D * tan_angle;
// Radius of opening at the surface y = 0
crater_lip_radius = interface_radius + depth_2D * tan_angle;

Point(1) = {0, r1, 0, dx};    // Far top point
Point(2) = {0, 0, 0, dx};     // Boundary circle center
Point(3) = {r1, 0, 0, dx};    // Far right point
Point(4) = {crater_lip_radius, 0, 0, dx};                         // Top point of crater
Point(5) = {crater_lip_radius, -depth_2D, 0, dx};             // Fillet center of crater
Point(6) = {interface_radius, -depth_2D, 0, dx};                // Bottom point of crater
Point(7) = {interface_radius, -depth_2D, 0, dx}; // Bottom outer point of conduit
Point(8) = {0, -depth_2D, 0, dx};    // Origin (along symmetry axis)

Circle(1)  = {1, 2, 3};        // First domain boundary r1
Line(2)    = {3, 4};           // Ocean floor outer
Line(3) = {4, 6};             // Jet slope
Line(5) = {6, 8};                // Inlet coupling boundary
Line(6)   = {8, 1};          // Symmetry line

// Define curve loop from curves
Curve Loop(1) = {3, 5, 6, 1, 2};

// Set planes with compatible (reversed) orientation
Plane Surface(1) = {-1};

// Set refinement from refinement curve
// Set refinement near origin
Field[3] = Distance;
// Field[3].CurvesList = {4};
Field[3].PointsList = {2};
// Distance-based refinement
Field[4] = MathEval;
Field[4].F = Sprintf("((F3/400)^2 + 1) * %g", dx_L1);
// Limit mesh size >= dx_L1 in magic box
Field[5] = Box;
Field[5].VIn  = dx_L2;
Field[5].VOut = dx;
Field[5].XMin = 0.0 * crater_lip_radius;
Field[5].XMax = 1.5 * crater_lip_radius;
Field[5].YMin = -depth_2D;
Field[5].YMax = 0.0;

// Set min field
Field[9] = Min;
Field[9].FieldsList = {4, 5};
Background Field = 9;

// Generate 2D mesh
Mesh 2;

// RefineMesh;
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";
OptimizeMesh "Laplace2D";

// Mesh export
Physical Curve("r1", 1) = {1};
Physical Curve("surface", 2) = {2};
Physical Curve("jetslope", 3) = {3};
Physical Curve("x2", 5) = {5};
Physical Curve("symmetry", 6) = {6};

Physical Surface("domain1") = {1};

Save Str(msh_target_name);