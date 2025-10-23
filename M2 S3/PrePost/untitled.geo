SetFactory("OpenCASCADE");

Point(1) = {0, 0, 0, 1.0};
Point(2) = {2, 0, 0, 1.0};
Point(3) = {2, 2, 0, 1.0};
Point(4) = {0.5, 1.5, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4}; 

Point(5) = {0.8, 0.3, 0, 1.0};
Point(6) = {1.5, 0.5, 0, 1.0};
Point(7) = {1.0, 0.8, 0, 1.0};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 5};

Line Loop(2) = {5, 6, 7};

Plane Surface(1) = {1, 2}; 