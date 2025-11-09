Merge "square.geo";

Transfinite Curve{1} = 20 Using Bump 0.2;
Transfinite Curve{2} = 20 Using Bump 0.2;
Transfinite Curve{3} = 20 Using Bump 0.2;
Transfinite Curve{4} = 20 Using Bump 0.2;

Transfinite Surface{1:4};


Lz =1;
out[] = Extrude {0,0,Lz} {Surface{1}; Layers{ {8,2}, {0.25,1} }; Recombine;}
Physical Volume("cube") = {out[0]};