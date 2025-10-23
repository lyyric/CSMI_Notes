import gmsh
import sys

gmsh.initialize()
gmsh.model.add("cube_with_cylindrical_hole")

L = 1.0
R = 0.2
lc = 0.05

box = gmsh.model.occ.addBox(-L/2, -L/2, -L/2, L, L, L)
cyl = gmsh.model.occ.addCylinder(0, 0, -L/2, 0, 0, L, R)

gmsh.model.occ.cut([(3, box)], [(3, cyl)], removeTool=True)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
