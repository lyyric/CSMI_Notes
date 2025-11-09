Merge "square.geo";
SetFactory("Built-in");

r = 0.4;
Point(10) = {0, 0, -dz, lc};

// 四个圆弧顶点（与对角线方向一致）
p1 = newp; Point(p1) = { r/sqrt(2),  r/sqrt(2), -dz, lc};
p2 = newp; Point(p2) = {-r/sqrt(2),  r/sqrt(2), -dz, lc};
p3 = newp; Point(p3) = {-r/sqrt(2), -r/sqrt(2), -dz, lc};
p4 = newp; Point(p4) = { r/sqrt(2), -r/sqrt(2), -dz, lc};

// 圆孔
Circle(11) = {p1,10,p2};
Circle(12) = {p2,10,p3};
Circle(13) = {p3,10,p4};
Circle(14) = {p4,10,p1};

// 删除旧面
Delete { Surface{1}; };

// 对角线（从角点连到孔边点）
Line(101) = {1, p3};
Line(102) = {3, p1};
Line(103) = {2, p4};
Line(104) = {4, p2};

// 四块区域
Curve Loop(21) = {1,102,-11,-104};
Plane Surface(21) = {21};

Curve Loop(22) = {2,103,-14,-102};
Plane Surface(22) = {22};

Curve Loop(23) = {3,101,-13,-103};
Plane Surface(23) = {23};

Curve Loop(24) = {4,104,-12,-101};
Plane Surface(24) = {24};

// 物理分组
Physical Surface("Part1") = {21};
Physical Surface("Part2") = {22};
Physical Surface("Part3") = {23};
Physical Surface("Part4") = {24};

Physical Curve("OuterBoundary") = {1,2,3,4};
Physical Curve("HoleBoundary") = {11,12,13,14};
Physical Curve("Diagonals") = {101,102,103,104};

// 网格自适应
Field[1] = Distance;
Field[1].NodesList = {p1,p2,p3,p4};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc*0.3;
Field[2].SizeMax = lc*2.0;
Field[2].DistMin = 0.2*r;
Field[2].DistMax = r;
Background Field = 2;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.RecombineAll = 1;
Mesh 2;
