import gmsh
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class RingConfig:
    """Configuration for ring geometry parameters."""
    r1: float  # Inner radius
    r2: float  # Outer radius
    h: float  # Height
    n: int  # Number of slits
    r_slit: float  # Radius of slit
    e_slit: float  # Radial width of slit
    angular_length: float  # Angular length of slit (degrees)
    name: str = "ring"  # Component name
    add_fillet: bool = False  # Add fillets to slits
    eps: float = 1e-3  # Tolerance for physical groups
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RingConfig':
        """Create RingConfig from dictionary."""
        return cls(**data)



class Ring:
    """Ring geometry component."""
    
    def __init__(self, config: RingConfig):
        """Initialize ring component.
        
        Args:
            config: Ring configuration parameters
        """
        self.config = config
        self.volume_ids = []
        self.physical_groups = {}
    
    def create_single_slit(self, angular_length_rad: float) -> List[Tuple]:
        """Create a single slit geometry.
        
        Args:
            angular_length_rad: Angular length in radians
            
        Returns:
            List of slit dimension tags
        """
        # Create inner and outer cylinders for slit
        slit_int = gmsh.model.occ.addCylinder(
            0, 0, -self.config.h / 4.0, 
            0.0, 0.0, 2 * self.config.h, 
            self.config.r_slit - self.config.e_slit / 2.0, 
            tag=-1, 
            angle=angular_length_rad
        )
        slit_ext = gmsh.model.occ.addCylinder(
            0, 0, -self.config.h / 4.0, 
            0.0, 0.0, 2 * self.config.h, 
            self.config.r_slit + self.config.e_slit / 2.0, 
            tag=-1, 
            angle=angular_length_rad
        )
        
        # Cut to create slit
        outDimTags, _ = gmsh.model.occ.cut(
            [(3, slit_ext)], 
            [(3, slit_int)], 
            removeObject=True, 
            removeTool=True
        )
        slit = outDimTags
        gmsh.model.occ.synchronize()
        
        # Center the slit
        gmsh.model.occ.rotate(slit, 0, 0, 0, 0, 0, 1, -angular_length_rad / 2)
        
        return slit
    
    def add_fillets_to_slit(self, slit: List[Tuple], angular_length_rad: float) -> List[Tuple]:
        """Add fillets to slit edges.
        
        Args:
            slit: Slit dimension tags
            angular_length_rad: Angular length in radians
            
        Returns:
            Modified slit with fillets
        """
        # Create first fillet
        fillet_0 = gmsh.model.occ.addCylinder(
            self.config.r_slit, 0, -self.config.h / 4.0, 
            0.0, 0.0, 2 * self.config.h, 
            self.config.e_slit / 2.0
        )
        gmsh.model.occ.rotate([(3, fillet_0)], 0, 0, 0, 0, 0, 1, -angular_length_rad / 2)
        
        # Create second fillet
        out = gmsh.model.occ.copy([(3, fillet_0)])
        fillet_1 = out[0][1]
        gmsh.model.occ.rotate([(3, fillet_1)], 0, 0, 0, 0, 0, 1, angular_length_rad)
        gmsh.model.occ.synchronize()
        
        # Fuse slit with fillets
        outDimTags, _ = gmsh.model.occ.fuse(
            slit, 
            [(3, fillet_0), (3, fillet_1)], 
            removeObject=True, 
            removeTool=True
        )
        gmsh.model.occ.synchronize()
        
        return outDimTags
    
    def generate(self) -> List[int]:
        """Generate the ring geometry.
        
        Returns:
            List of volume IDs created
        """
        print(f'\n=== Generating Ring: {self.config.name} ===')
        
        # Create base cylinder
        cyl_int = gmsh.model.occ.addCylinder(
            0, 0, 0, 
            0, 0, self.config.h, 
            self.config.r1
        )
        cyl_ext = gmsh.model.occ.addCylinder(
            0, 0, 0, 
            0, 0, self.config.h, 
            self.config.r2
        )
        
        outDimTags, _ = gmsh.model.occ.cut(
            [(3, cyl_ext)], 
            [(3, cyl_int)], 
            removeObject=True, 
            removeTool=True
        )
        cyl = outDimTags
        gmsh.model.occ.synchronize()
        
        # Create slits
        angular_length_rad = self.config.angular_length * math.pi / 180.0
        theta = 2 * math.pi / self.config.n
        
        # Create base slit
        slit = self.create_single_slit(angular_length_rad)
        print('slit:', slit)
        
        # Add fillets if requested
        if self.config.add_fillet:
            slit = self.add_fillets_to_slit(slit, angular_length_rad)
        
        # Create and rotate multiple slits
        slits = []
        for i in range(self.config.n):
            out = gmsh.model.occ.copy(slit)
            gmsh.model.occ.rotate(out, 0, 0, 0, 0, 0, 1, i * theta)
            slits.append(out[0])
        
        # Cut slits from cylinder
        outDimTags, _ = gmsh.model.occ.cut(
            cyl, 
            slits, 
            removeObject=True, 
            removeTool=True
        )
        ring = outDimTags

        gmsh.model.occ.remove(slit, True)
        gmsh.model.occ.synchronize()
        
        volume_ids = [ring[0][1]]
        print(f'  Created volumes: {volume_ids}')
        
        return volume_ids
    
    def create_physical_groups(self, volume_ids: List[int], prefix=""):
        """Create physical groups for the ring."""
        if len(volume_ids) == 0:
            print(f'  Warning: No volumes for {self.config.name}')
            return
        
        if prefix != "":
            prefix += "_"

        print(f'\n=== Creating Ring Physical Groups for {self.config.name} ===')
        gmsh.model.addPhysicalGroup(3, volume_ids, name=self.config.name)

        # TODO create surface physical groups
        eps = self.config.eps
        self.create_boundary_groups(volume_ids, prefix=prefix)
        
    def create_boundary_groups(self, volume_ids: List[int], prefix=""):
        """Create physical groups for ring boundaries."""
        eps = self.config.eps
        r1 = self.config.r1
        r2 = self.config.r2

        # Get bounding box for z-coordinates
        bbox = gmsh.model.occ.getBoundingBox(3, volume_ids[0])
        zmin, zmax = bbox[2], bbox[5]

        # --- Top and bottom faces (V0 / V1) ---
        # Bottom surface (z ≈ zmin)
        V0 = gmsh.model.getEntitiesInBoundingBox(
            -r2 - eps, -r2 - eps, zmin - eps,
             r2 + eps,  r2 + eps, zmin + eps,
            2
        )
        if len(V0) > 0:
            print(f'Created V0 group: {len(V0)} surfaces (V0={V0})')
            gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in V0], name=f"{prefix}V0")

        # Top surface (z ≈ zmax)
        V1 = gmsh.model.getEntitiesInBoundingBox(
            -r2 - eps, -r2 - eps, zmax - eps,
             r2 + eps,  r2 + eps, zmax + eps,
            2
        )
        if len(V1) > 0:
            print(f'Created V1 group: {len(V1)} surfaces (V1={V1})')
            gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in V1], name=f"{prefix}V1")


        # --- Inner cylindrical surface (Rint) ---
        rint = gmsh.model.getEntitiesInBoundingBox(
            -r1 - eps, -r1 - eps, zmin - eps,
             r1 + eps,  r1 + eps, zmax + eps,
            2
        )
        if len(rint) > 0:
            gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in rint], name=f"{prefix}Rint")
            print(f'Found {len(rint)} surfaces for Rint (rint={rint})')


        # --- Cooling slit cylindrical surfaces (Rslit) ---
        # Outer slit boundary
        r = self.config.r_slit + self.config.e_slit / 2. + eps
        rslit_in = gmsh.model.getEntitiesInBoundingBox(
            -r, -r, zmin - eps,
             r,  r, zmax + eps,
            2
        )
        # Inner slit boundary
        r = self.config.r_slit - self.config.e_slit / 2. - eps
        rslit_ext = gmsh.model.getEntitiesInBoundingBox(
            -r, -r, zmin - eps,
             r,  r, zmax + eps,
            2
        )
        rslit = rslit_in + rslit_ext
        if len(rslit) > 0:
            gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in rslit], name=f"{prefix}Coolingslit")
            print(f'Found {len(rslit)} surfaces for Coolingslit (rslit={rslit})')


        # --- Outer cylindrical surface (Rext) ---
        rext = gmsh.model.getEntitiesInBoundingBox(
            -r2 - eps, -r2 - eps, zmin - eps,
             r2 + eps,  r2 + eps, zmax + eps,
            2
        )
        # Avoid double-counting with V0/V1/Rint/Rslit
        exclude = set(rint + V0 + V1 + rslit)
        rext_filtered = [(dim, tag) for (dim, tag) in rext if (dim, tag) not in exclude]

        if len(rext_filtered) > 0:
            gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in rext_filtered], name=f"{prefix}Rext")
            print(f'Found {len(rext_filtered)} surfaces for Rext (rext={rext_filtered})')


    def generate_mesh(self, mesh_size: Optional[float] = None):
        """Generate mesh for the entire insert.
        
        Args:
            mesh_size: Global mesh size (if None, auto-calculated)
        """
        print("\n=== Generating Mesh ===")
        
        if mesh_size is None:
            # Auto-calculate mesh size based on first component
            mesh_size = abs(self.config.r2 - self.config.r1) / 3.

        print(f"Mesh size: {mesh_size:.4f}")
        
        # Set mesh size for all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        
        # Generate 3D mesh
        print('Generating 3D mesh...')
        OUTPUT_MESH_FILE = self.config.name + ".msh"
        gmsh.model.mesh.generate(3)
        gmsh.write(OUTPUT_MESH_FILE)
        print(f'Mesh written to: {OUTPUT_MESH_FILE}')

def main():
    """Main function to orchestrate the helix generation process."""
    print("=" * 60)
    print("Helix Geometry Generator")
    print("=" * 60)
    
    from insert import parse_arguments, load_config

    try:
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.model.add("ring")
        
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration from JSON file
        data = load_config(args.config)
        
        # Parse components
        ringconfig = RingConfig.from_dict(data)
        ring = Ring(ringconfig)
        
        # Generate geometry
        ring_ids = ring.generate()
        
        # Create physical groups
        ring.create_physical_groups(ring_ids)
        
        # Generate mesh if requested
        if args.mesh:
            ring.generate_mesh(args.mesh_size)
        
        print("\n" + "=" * 60)
        print("Ring generation completed successfully")
        print("=" * 60)
        
        # Show GUI if not suppressed
        if not args.nopopup:
            gmsh.fltk.run()
            
    except Exception as e:
        print(f"\nError during ring generation: {e}")
        raise
    finally:
        gmsh.finalize()  # Uncomment if you want to finalize Gmsh


if __name__ == "__main__":
    main()
