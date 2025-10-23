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

print("=== All entities in model ===")
entities = gmsh.model.getEntities()
for e in entities:
    print(f"Entity {e} of type {gmsh.model.getType(e[0], e[1])}")


volumes = gmsh.model.getEntities(dim=3)
print("\n=== Volume entities ===")
print(volumes)


boundaries = gmsh.model.getBoundary(volumes)
print("\n=== Boundary surfaces of the cube ===")
for b in boundaries:
    print("Surface:", b)


z_tol = 1e-3
faces_top = gmsh.model.getEntitiesInBoundingBox(
    -L/2, -L/2, L/2 - z_tol, L/2, L/2, L/2 + z_tol, dim=2)
print("\n=== Surfaces near top (z≈+L/2) ===")
print(faces_top)

print("\n=== Mesh generation ===")

gmsh.model.mesh.generate(3)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
