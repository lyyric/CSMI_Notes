CXX      = g++-15
CXXFLAGS = -O3 -march=native -fopenmp -std=c++11

SRC  = grid2d.cpp leapfrog2d.cpp
HDR  = grid2d.hpp

all: leapfrog_std leapfrog_std_v2 leapfrog_tiled tile16 tile32 tile64


# -----------------------------
# Standard version
# -----------------------------
leapfrog_std: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

# -----------------------------
# Standard version (OpenMP V2)
# compile with -DOMP_V2
# -----------------------------
leapfrog_std_v2: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DOMP_V2 -o $@ $(SRC)

# -----------------------------
# Tiled version (layout tiling)
# compile with -DTILING
# -----------------------------
leapfrog_tiled: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DTILING -o $@ $(SRC)

# -----------------------------
# Loop tiling version (TP2)
# compile with -DLOOP_TILING and -DTILE_SIZE=...
# -----------------------------
tile16: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DLOOP_TILING -DTILE_SIZE=16 -o $@ $(SRC)

tile32: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DLOOP_TILING -DTILE_SIZE=32 -o $@ $(SRC)

tile64: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -DLOOP_TILING -DTILE_SIZE=64 -o $@ $(SRC)


clean:
	rm -f leapfrog_std leapfrog_std_v2 leapfrog_tiled plot.dat plot.gp
