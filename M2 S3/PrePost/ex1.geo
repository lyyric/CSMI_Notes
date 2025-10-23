SetFactory("Built-in");

// ---- Parameters ----
lc = DefineNumber[0.02, Name "Parameters/lc"];
Lx = DefineNumber[2.5, Name "Parameters/Lx"];
Ly = DefineNumber[0.41, Name "Parameters/Ly"];
R  = DefineNumber[0.05, Name "Parameters/R"];
x_c = 0.2; y_c = Ly/2;

// ---- Rectangle domain ----
Point(1) = {0, 0, 0, lc};
Point(2) = {Lx, 0, 0, lc};
Point(3) = {Lx, Ly, 0, lc};
Point(4) = {0, Ly, 0, lc};

Line(1) = {1, 2}; Line(2) = {2, 3};
Line(3) = {3, 4}; Line(4) = {4, 1};

// ---- Circular cylinder ----
Point(10) = {x_c, y_c, 0, lc};
Point(11) = {x_c+R, y_c, 0, lc};
Point(12) = {x_c, y_c+R, 0, lc};
Point(13) = {x_c-R, y_c, 0, lc};
Point(14) = {x_c, y_c-R, 0, lc};

Circle(11) = {11,10,12};
Circle(12) = {12,10,13};
Circle(13) = {13,10,14};
Circle(14) = {14,10,11};

// ---- Tail (elastic beam) ----
L_tail = 0.35; H_tail = 0.02;
Point(20) = {x_c+R, y_c-H_tail/2, 0, lc};
Point(21) = {x_c+R+L_tail, y_c-H_tail/2, 0, lc};
Point(22) = {x_c+R+L_tail, y_c+H_tail/2, 0, lc};
Point(23) = {x_c+R, y_c+H_tail/2, 0, lc};

Line(21) = {20,21}; Line(22) = {21,22};
Line(23) = {22,23}; Line(24) = {23,20};
Curve Loop(3) = {21,22,23,24};
Plane Surface(3) = {3};

// ---- Flow domain with hole ----
Curve Loop(1) = {1,2,3,4};
Curve Loop(2) = {11,12,13,14};
Plane Surface(1) = {1, -2, -3};

// ---- Physicals ----
Physical Surface("Fluid") = {1};
Physical Curve("Inlet") = {4};
Physical Curve("Outlet") = {2};
Physical Curve("Walls") = {1,3};
Physical Curve("Cylinder") = {11,12,13,14};
Physical Surface("Tail") = {3};
