SetFactory("OpenCASCADE");

Printf("Loading H1");
Include "H1.geo";
H1 = cu;
H1_isolant = isolant;
For i In {0 : #H1()-1}
  Printf("H1[%g]=%g", i, H1(i));
EndFor
H1_dz = dz;

Printf("Loading H1");
Include "H2.geo";
H2 = cu;
H2_isolant = isolant;
For i In {0 : #H2()-1}
  Printf("H2[%g]=%g", i, H2(i));
EndFor

Printf("Loading Ring");
Include "ring.geo";
R = ring;

//Translate ring
Translate {0, 0, H1_dz/2.} { Volume{R}; }

//Make fragment
Printf("Creating insert:");
insert() = BooleanFragments{ Volume{H1, H1_isolant, H2, H2_isolant}; } { Volume{R}; };
For i In {0 : #insert()-1}
  Printf("insert[%g]=%g", i, insert(i));
EndFor

Physical Volume("H1") = {insert[0]};
Physical Volume("H2") = {insert[2]};
Physical Volume("R") = {insert[#insert()-1]};

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

// Whole domain surface
bnd() = Abs(Boundary{ Volume{insert[0]}; });
Physical Surface("Bord_H1") = bnd();
bnd() = Abs(Boundary{ Volume{insert[2]}; });
Physical Surface("Bord_H2") = bnd();
bnd() = Abs(Boundary{ Volume{insert[#insert()-1]}; });
Physical Surface("Bord_R") = bnd();
