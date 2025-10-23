import gmsh
import sys

gmsh.initialize()
gmsh.model.add("square")

Lc = 1e-1
Lx = Ly = 1.0

p1 = gmsh.model.geo.addPoint(-Lx, -Ly, 0, Lc)
p2 = gmsh.model.geo.addPoint(Lx, -Ly, 0, Lc)
p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, Lc)
p4 = gmsh.model.geo.addPoint(-Lx, Ly, 0, Lc)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surface = gmsh.model.geo.addPlaneSurface([loop])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
