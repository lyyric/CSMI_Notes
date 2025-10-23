SetFactory("OpenCASCADE");


Box(1) = {0, 0, 0, 1, 1, 1};


Cylinder(2) = {0.5, 0.5, 0, 0, 0, 1, 0.2}; 



BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }


Physical Volume("CubeWithHole") = {1};
