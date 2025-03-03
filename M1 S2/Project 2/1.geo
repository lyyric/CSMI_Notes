// Gmsh project created on Fri Feb 23 10:54:19 2024
h=0.1;
//+
SetFactory("Built-in");
//+
Point(1) = {0, 0, 0, h};
//+
Point(2) = {2, 0.4, 0, h/10};
//+
Point(3) = {1, 1, 0, h/5};
//+
Point(4) = {0, 1, 0, h};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
//Plane Surface(1) = {1};
//+
Point(5) = {0.5, 0.5, 0, h/5};
//+
Point(6) = {0.6, 0.5, 0, h/5};
//+
Point(7) = {0.4, 0.5, 0, h/5};
//+
Circle(5) = {6, 5, 7};
//+
Circle(6) = {7, 5, 6};
//+
//+
Curve Loop(2) = {5, 6};
//+
Plane Surface(1) = {1,2};
Plane Surface(2) = {2};
//+
Physical Curve("Gamma_D", 7) = {4, 1};
//+
Physical Curve("Gamma_N", 8) = {3, 2};
//+
Physical Surface("Omega_1", 9) = {1};
//+
Physical Surface("Omega_2", 10) = {2};
