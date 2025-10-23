SetFactory("OpenCASCADE");

Lc = 0.1;
Lc_h = 0.05;
R = 0.1;
Lz = 0.25;

Point(1) = {-0.5, -0.5, 0, Lc};
Point(2) = { 0.5, -0.5, 0, Lc};
Point(3) = { 0.5,  0.5, 0, Lc};
Point(4) = {-0.5,  0.5, 0, Lc};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(1) = {1,2,3,4};

O = newp; Point(O) = {0,0,0,Lc_h};
p5 = newp; Point(p5) = {R,0,0,Lc_h};
p6 = newp; Point(p6) = {0,R,0,Lc_h};
p7 = newp; Point(p7) = {-R,0,0,Lc_h};
p8 = newp; Point(p8) = {0,-R,0,Lc_h};
c5 = newl; Circle(c5) = {p5,O,p6};
c6 = newl; Circle(c6) = {p6,O,p7};
c7 = newl; Circle(c7) = {p7,O,p8};
c8 = newl; Circle(c8) = {p8,O,p5};
Curve Loop(2) = {c5,c6,c7,c8};

Plane Surface(1) = {1,-2};

out[] = Extrude {0,0,Lz} { Surface{1}; };

Physical Surface("Bottom") = {out[0]};
Physical Surface("Top") = {out[1]};
Physical Surface("Sides") = {out[2:5]};
Physical Volume("CubeWithHole") = {out[6]};
