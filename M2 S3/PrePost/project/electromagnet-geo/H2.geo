// -----------------------------------------------------------------------------
//
//  Gmsh GEO tutorial 19
//
//  Thrusections, fillets, pipes, mesh size from curvature
//
// -----------------------------------------------------------------------------

SetFactory("OpenCASCADE");

r1 = 25.1;
r2 = 30.7;
dz = 2*150;
cut = 2;
eps = 0.4; // shall not exceed the distance between rint and previous helix, nor the distance between rext and the next helix
nturns = 6;
pitch = 18.03646917296748;
sens = 1;

// We define the shape we would like to extrude along the spline:
s=news; Rectangle(s) = {r1-eps, -cut/2., 0, r2-r1+2*eps, cut};
Rotate {{1, 0, 0}, {0, 0, 0}, Pi/2} { Surface{s}; }
bnd() = Abs(Boundary{ Surface{s}; });
/******
Printf('Scut=%g', s);
Printf('Bord: %g', #bnd()-1);
For i In {0: #bnd()-1}
  Printf("Bord[%g]=%g", i, bnd[i]);
EndFor  
******/

// OpenCASCADE also allows general extrusions along a smooth path.
// Let's first define a spline curve:
npts = 60 * nturns;
r = r1 + (r2-r1) /2.;
h = pitch * nturns;
//points = {};
sections = {};
For i In {0 : npts}
  theta = i * 2*Pi*nturns/npts;
  // Printf("helix[%g]: theta=%g", i, theta);
  p = newp; Point(p) = {r * Cos(sens*theta), r * Sin(sens*theta), -h/2. + i * h/npts};
  //points[i] = p;
  ns() = Rotate {{0, 0, 1}, {0, 0, 0}, sens*theta} { Duplicata{Surface{s};} };
  nns() = Translate {0, 0, -h/2. + i * h/npts} { Surface{ns()}; };
  bnd() = Abs(Boundary{ Surface{nns}; });
  /*****
  For j In {0: #bnd()-1}
    Printf("Surface[%g]: Bord[%g]=%g", i, j, bnd[j]);
  EndFor  
  ***/
  nbnd = newl; Curve Loop(nbnd) = {bnd[0]:bnd[#bnd()-1]};
  // Printf('Surface: %g, Bord: %g', nns, nbnd);
  sections[i] = nbnd; // nns;
  Delete { Surface{nns};}
EndFor
Printf('sections=%g (from %g to %g)', #sections(), sections[0], sections[#sections()-1]);

hcut = news; Ruled ThruSections(hcut) = {sections[]};
Printf('hcut=%g', hcut);
Delete { Curve{sections[]}; }

// TODO add cylnder at the beginning an end of hcut
wire_begin = news; Cylinder(wire_begin) = {r1-eps, 0, -nturns*pitch/2., (r2-r1)+2*eps, 0., 0, cut/2.};
wire_end = news; Cylinder(wire_end) = {r1-eps, 0, +nturns*pitch/2.+2*cut-cut/2., (r2-r1)+2*eps, 0., 0, 2*cut};
nhcut = BooleanUnion { Volume{hcut}; Delete;} {Volume{wire_begin, wire_end}; Delete;};

int = news; Cylinder(int) = {0, 0, -dz/2., 0, 0, dz, r1};
ext = news; Cylinder(ext) = {0, 0, -dz/2., 0, 0, dz, r2};
cyl = BooleanDifference { Volume{ext}; } { Volume{int}; };
Recursive Delete { Volume {ext, int}; }

helix() = BooleanFragments{ Volume{cyl}; Delete;} { Volume{nhcut}; Delete; };
Printf("Helix: %g", #helix());
For i In {0 : #helix()-1}
   Printf("helix[%g]=%g", i, helix(i));
EndFor
Recursive Delete { Volume{helix(2), helix(3)};}
Recursive Delete { Surface{s};}

cu = helix[0];
isolant = helix[1];

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


Physical Volume("Cu") = {cu};
Physical Volume("Isolant") = {isolant};

f() = Abs(Boundary{ Volume{helix(0)}; });
For i In {0 : #f()-1}
   Printf("helix[%g]=%g", i, f(i));
EndFor

// Whole domain surface
bnd() = Abs(Boundary{ Volume{cu}; });
Physical Surface(80) = bnd();

***/
