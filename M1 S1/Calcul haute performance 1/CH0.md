Calcul haute performance 1
Chabannes Vincent
moodle

mpicxx main.cpp

mpic++ main.cpp -o main
mpiexec -n 4 ./main
mpiexec -n 20 --oversubscribe ./main
