Merge "square.geo";

// Distance Field
Field[1] = Distance;
Field[1].CurvesList = {1,3};
Field[1].Sampling = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc / 10;
Field[2].SizeMax = lc;
Field[2].DistMin = 0.005;
Field[2].DistMax = 0.025;
Background Field = 2;