// ---------------------------------------
//
// Create a ring with slented slits and filltes
//
// ---------------------------------

SetFactory("OpenCASCADE");

r1 = 19.3;
r2 = 30.7;
h = 20;

n = 4; // number of slit
r_slit = (24.2 + 25.1)/2.; // radius of slit
e_slit = (25.1-24.2); // radial width of slit
angular_length = 30 * Pi / 180.; // angular length of slit
theta = 2 * Pi / n;

int =  news; Cylinder(int) = {0, 0, 0., 0, 0, h, r1};
ext = news; Cylinder(ext) = {0, 0, 0., 0, 0, h, r2};
cyl = BooleanDifference { Volume{ext}; } { Volume{int}; };
Recursive Delete { Volume {ext, int}; }

slits = {};
slit_int = news; Cylinder(slit_int) = {0, 0, -h/4., 0., 0., 2*h, r_slit - e_slit/2., angular_length};
slit_ext = news; Cylinder(slit_ext) = {0, 0, -h/4., 0., 0., 2*h, r_slit + e_slit/2., angular_length};
fillet_0 = news; Cylinder(fillet_0) = {r_slit, 0, -h/4., 0., 0., 2*h, e_slit/2.}; 
slit = BooleanDifference { Volume{slit_ext}; } { Volume{slit_int}; };
Recursive Delete { Volume {slit_ext, slit_int}; }

// rotate slit
Rotate {{0, 0, 1}, {0, 0, 0}, -angular_length/2.} { Volume{slit};}
Rotate {{0, 0, 1}, {0, 0, 0}, -angular_length/2.} { Volume{fillet_0};}
fillet_1 = Rotate {{0, 0, 1}, {0, 0, 0}, angular_length} { Duplicata{Volume{fillet_0};} };
cslit = BooleanUnion { Volume{slit}; } { Volume{fillet_0, fillet_1}; };
Recursive Delete { Volume {slit, fillet_0, fillet_1}; }
slits[0] = cslit;

//  create all slits
For i In {1 : n}
    nslit() = Rotate {{0, 0, 1}, {0, 0, 0}, theta} { Duplicata{Volume{cslit};} };
    slits[i] = nslit;
    cslit = nslit;
EndFor

ring = BooleanDifference { Volume{cyl}; } { Volume{slits[0]:slits[#slits()-1]}; };
Recursive Delete { Volume {cyl, slits[0]:slits[#slits()-1]}; }

/****
// We delete the source surface, and increase the number of sub-edges for a
// nicer display of the geometry:
// Delete{ Point{points}; }
Geometry.NumSubEdges = 1000;

// We can activate the calculation of mesh element sizes based on curvature
// (here with a target of 20 elements per 2*Pi radians):
Mesh.MeshSizeFromCurvature = 40;

// We can constraint the min and max element sizes to stay within reasonnable
// values (see `t10.geo' for more details):
Mesh.MeshSizeMin = 0.1;
Mesh.MeshSizeMax = 1;


Physical Volume("Cu") = {ring};

f() = Abs(Boundary{ Volume{ring(0)}; });
For i In {0 : #f()-1}
   Printf("ring[%g]=%g", i, f(i));

// Whole domain surface
bnd() = Abs(Boundary{ Volume{ring}; });
Physical Surface(80) = bnd();
***/