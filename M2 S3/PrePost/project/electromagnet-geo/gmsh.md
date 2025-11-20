# An electromagnet with `Gmsh`

The files are just given as a reference.
They provide a way to check your work and help you create the geometry using `Gmsh` Python API

## The geometry

The electromagnet consists of:

* `H1.geo` : a cylindrical tube with an helical cut (hcut)
* `H2.geo` : a second ...................................
* `ring.geo` : a connection ring between H1 and H2 that allows a water flow in between H1 and H2 (hence the coolingslits)

## To create the geometry

```bash
gmsh insert.geo
```

## To create a mesh

```bash
gmsh insert.geo  -clcurv 20 -algo del2d -rand 1.e-12 -algo hxt -3
```

The `-rand 1.e-12` is needed for some algorithmic details in `hxt` 3D tetra mesher.

[NOTE]
====
The `Physicals` are not properly defined in `insert.geo`.
We only define `Physicals` for the volumic domains and for the boundaries of each component (eg. H1).
====
