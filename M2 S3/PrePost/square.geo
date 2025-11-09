SetFactory("Built-in");

lc = 0.1;
dx = 1;
dy = 1;
dz = 1;

// create square
Point(1) = {-dx, -dy, -dz, lc};
Point(2) = {dx, -dy, -dz, lc};
Point(3) = {dx, dy, -dz, lc};
Point(4) = {-dx, dy, -dz, lc};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1:4};
Plane Surface(1) = {1};

Physical Surface("Top") = {1};
Physical Curve("Ox") = {1};