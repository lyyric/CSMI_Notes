Merge "square.geo";
Lz =1;
out[] = Extrude {0,0,Lz} {Surface{1}; Layers{ {8,2}, {0.25,1} };};
Physical Volume("cube") = {out[0]};

