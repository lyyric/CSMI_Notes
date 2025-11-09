import gmsh
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class HelixConfig:
    """Configuration for helix geometry parameters."""
    r1: float  # Inner radius
    r2: float  # Outer radius
    z1: float # base
    z2: float  # top
    cut: float  # Cut width
    eps: float  # Tolerance/epsilon
    nturns: int  # Number of helix turns TODO nturns is an array for variable helical cut
    pitch: float  # Helix pitch TODO pitch is an array for variable helical cut 
    npts: int = 60  # Number of points along helix
    start_hole_diameter: float = 1.6  # EDM wire hole diameter
    z_offset: float = 0.0  # Vertical offset for positioning
    add_start_hole: bool = False # Add EDM start hole
    sens: int = 1 # Orientation of Helical cut
    name: str = "helix"  # Component name
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HelixConfig':
        """Create HelixConfig from dictionary."""
        return cls(**data)


class Helix:
    """Helix geometry component."""
    
    def __init__(self, config: HelixConfig, add_start_hole: bool = False):
        """Initialize helix component.
        
        Args:
            config: Helix configuration parameters
            add_start_hole: Whether to add EDM wire start hole
        """
        self.config = config
        self.add_start_hole = add_start_hole
        self.volume_ids = []
        self.physical_groups = {}
        
    def create_helix_sections(self) -> Tuple[List[int], float]:
        """Create the helix cross-sections along the spline.
        
        Returns:
            Tuple of (sections list, helix height)
        """
        h = self.config.pitch * self.config.nturns
        
        # Define shape to extrude along spline
        s = gmsh.model.occ.addRectangle(
            self.config.r1 - self.config.eps, 
            -self.config.cut / 2., 
            0, 
            self.config.r2 - self.config.r1 + 2 * self.config.eps, 
            self.config.cut
        )
        gmsh.model.occ.rotate([(2, s)], 0, 0, 0, 1, 0, 0, math.pi / 2.)
        # gmsh.model.occ.synchronize()
        
        sections = []
        for i in range(self.config.npts + 1):
            theta = i * 2 * math.pi * self.config.nturns / self.config.npts
            z = -h / 2. + i * h / self.config.npts + self.config.z_offset
            
            # Copy, rotate and translate section
            ov = gmsh.model.occ.copy([(2, s)])
            # print('ov:', ov, end=" --> ")
            gmsh.model.occ.rotate(ov, 0, 0, 0, 0, 0, 1, self.config.sens*theta)
            gmsh.model.occ.translate(ov, 0, 0, z)
            gmsh.model.occ.synchronize()
            
            curvelooptags, curvetags = gmsh.model.occ.getCurveLoops(ov[0][1])
            # print('curvetags:', curvetags, len(curvetags), end=" --> ", flush=True)
            wire = gmsh.model.occ.addCurveLoop(curvetags[0])
            # print('wire:', wire, flush=True)

            sections.append(wire)
            gmsh.model.occ.remove(ov, False)
            # gmsh.model.occ.synchronize()
        
        # Clean up base section
        gmsh.model.occ.remove([(2, s)], True)
        # gmsh.model.occ.synchronize()
        
        return sections, h
    
    def add_edm_start_hole(self, hcut) -> Tuple[List, Optional[Tuple]]:
        """Add EDM wire start hole to helix geometry.
        
        Args:
            hcut: Current helix cut geometry
            
        Returns:
            Tuple of (modified hcut, bounding box)
        """
        print(f'  Adding EDM wire start hole to {self.config.name}', flush=True)
        
        h = self.config.pitch * self.config.nturns
        d = self.config.start_hole_diameter
        _z = -self.config.sens*h/2. + self.config.sens*(d/2 - self.config.cut/2.)
        _z += self.config.z_offset
        _r0 = self.config.r1 - 4 * self.config.eps
        _r1 = self.config.r2 - self.config.r1 + 2 * 4 * self.config.eps
        start = gmsh.model.occ.addCylinder(
            _r0, 
            0, 
            _z , 
            _r1, 
            0, 
            0, 
            d / 2.
        )
        ## TODO h = nturns * pitch, need to rotate if nturns in not integer ##
        ## BPside or HPside depending of Helix sens

        # Fragment and fuse with helical cut
        outDimTags, _ = gmsh.model.occ.fragment(
            hcut, 
            [(3, start)], 
            removeObject=True, 
            removeTool=True
        )
        
        outDimTags, _ = gmsh.model.occ.fuse(
            [outDimTags[0]], 
            [outDimTags[i] for i in range(1, len(outDimTags))], 
            removeObject=True, 
            removeTool=True
        )
        ncut = outDimTags[0]
        # gmsh.model.occ.synchronize()
        
        hcut_bbox = gmsh.model.occ.getBoundingBox(3, ncut[1])
        
        return [ncut]

    def add_edm_hole(self, hcut: List) -> Tuple[List, Optional[Tuple]]:
        """Add EDM wire hole to helix geometry.
        
        Args:
            hcut: Current helix cut geometry
            
        Returns:
            Tuple of (modified hcut, bounding box)
        """
        print(f'  Adding EDM wire hole to {self.config.name}', flush=True)
        
        h = self.config.pitch * self.config.nturns
        d = self.config.start_hole_diameter
        _z = self.config.sens * h/2.
        _z += self.config.z_offset
        _r0 = self.config.r1 - 4 * self.config.eps
        _r1 = self.config.r2 - self.config.r1 + 2 * 4 * self.config.eps
        start = gmsh.model.occ.addCylinder(
            _r0, 
            0, 
            _z, 
            _r1, 
            0, 
            0, 
            self.config.cut / 2.
        )
        ## TODO h = nturns * pitch, need to rotate if nturns in not integer ##
        ## BPside or HPside depending of Helix sens

        # Fragment and fuse with helical cut
        outDimTags, _ = gmsh.model.occ.fragment(
            hcut, 
            [(3, start)], 
            removeObject=True, 
            removeTool=True
        )
        
        outDimTags, _ = gmsh.model.occ.fuse(
            [outDimTags[0]], 
            [outDimTags[i] for i in range(1, len(outDimTags))], 
            removeObject=True, 
            removeTool=True
        )
        ncut = outDimTags[0]
        # gmsh.model.occ.synchronize()
        
        hcut_bbox = gmsh.model.occ.getBoundingBox(3, ncut[1])
        
        return [ncut]
    
    def generate(self, ignore_ids: List[int] = []) -> List[int]:
        """Generate the helix geometry.
        
        Returns:
            List of volume IDs created
        """
        print(f'\n=== Generating Helix: {self.config.name} ===')
        
        # Create helix sections
        sections, h = self.create_helix_sections()
        
        # Create solid helix through sections
        hcut = gmsh.model.occ.addThruSections(
            sections, 
            makeSolid=True, 
            makeRuled=True, 
            continuity="G1", 
            parametrization="IsoParametric", 
            smoothing=False
        )
        # print(f"hcut={hcut}", flush=True)
        
        
        # Create base cylinder (annular region)
        ################ TO FILL IN THE ELLIPSES WITH PROPER ARGUMENTS ################
        dz = abs(self.config.z2 - self.config.z1)
        cint = gmsh.model.occ.addCylinder(0, 0, self.config.z1, 0, 0, dz, self.config.r1)
        cext = gmsh.model.occ.addCylinder(0, 0, self.config.z1, 0, 0, dz, self.config.r2)
        outDimTags, _ = gmsh.model.occ.cut(
            [(3, cext)],
            [(3, cint)],
            removeObject=True,
            removeTool=True
        )
        cyl = outDimTags[0]
        gmsh.model.occ.synchronize()

        
        # Add EDM wire start hole if requested
        if self.add_start_hole:
            hcut = self.add_edm_start_hole(hcut)
            hcut = self.add_edm_hole(hcut)
        
        
        # Fragment geometry to create separate volumes
        # print(f"cyl={cyl}, hcut={hcut}", flush=True)
        ################ TO FILL IN THE ELLIPSES WITH PROPER ARGUMENTS ################
        outDimTags, outDimTagsMap = gmsh.model.occ.fragment(
            [cyl],
            hcut if isinstance(hcut, list) else [hcut]
        )
        gmsh.model.occ.synchronize()

        
        cyl_children_id = [dimtag[1] for dimtag in outDimTagsMap[0]]
        # print(f'  Cylinder children IDs: {cyl_children_id}', flush=True)

        # Remove entities not belonging to cylinder
        volume_ids = []
        entities = gmsh.model.getEntities(3)
        for entity in entities:
            if entity[1] not in ignore_ids:
                if entity[1] not in cyl_children_id:
                    # print(f'  Removing entity not in cylinder: {entity}', flush=True)
                    gmsh.model.occ.remove([entity], True)
                else:
                    volume_ids.append(entity[1])
        
        for section in sections:
            gmsh.model.occ.remove([(1, section)], True)
        # gmsh.model.occ.synchronize()

        gmsh.model.occ.synchronize()
        # print(f'  Created volumes: {volume_ids}', flush=True)
        
        return volume_ids
        
    
    def create_physical_groups(self, volume_ids: List[int], prefix=""):
        """Create physical groups for the helix."""
        if len(volume_ids) < 2:
            print(f'  Warning: Expected 2 volumes for {self.config.name}, got {len(self.volume_ids)}')
            return
        
        print(f'\n=== Creating Helix Physical Groups for {self.config.name} ===')
        
        if prefix != "":
            prefix += "_"

        # Get boundaries for each volume
        bcs = {}
        for vol_id in volume_ids:
            boundaries = gmsh.model.getBoundary(
                [(3, vol_id)], 
                combined=False, 
                oriented=False, 
                recursive=False
            )
            bcs[vol_id] = sorted([e[1] for e in boundaries])
            print(f'Volume {vol_id}: {len(boundaries)} boundaries')
        
        # Find interface between Cu and Glue
        # WATCHOUT assume only two volumes - 1srt one is Cu WATCHOUT
        interface = list(set(bcs[volume_ids[0]]) & set(bcs[volume_ids[1]]))
        print(f'Interface boundaries: {len(interface)}')
        
        # Classify boundary types
        bcs_type = self.classify_boundaries(volume_ids, bcs, interface)
        
        gmsh.model.occ.synchronize()
        
        # Create volume physical groups
        gmsh.model.addPhysicalGroup(3, [volume_ids[0]], name=f"{prefix}Cu")
        gmsh.model.addPhysicalGroup(3, [volume_ids[1]], name=f"{prefix}Glue")
        print('Created volume physical groups: Cu, Glue')
        
        # Create interface physical group
        gmsh.model.addPhysicalGroup(2, interface, name=f"{prefix}Interface")
        print('Created interface physical group')
        
        # Create boundary physical groups
        eps = self.config.eps
        bcs_names = self.create_boundary_groups(volume_ids, bcs, bcs_type, interface, prefix)
        
        return bcs_names


    def classify_boundaries(self, volume_ids: List[int], bcs: Dict, interface: List[int]) -> Dict:
        """Classify boundaries by their geometric type.
        
        Args:
            volume_ids: List of volume IDs
            bcs: Boundary conditions dictionary
            interface: List of interface boundary IDs
            
        Returns:
            Dictionary mapping volume IDs to boundary types
        """
        bcs_type = {}
        for vol in volume_ids:
            bcs_type[vol] = {}
            for bc_id in bcs[vol]:
                if bc_id not in interface:
                    btype = gmsh.model.getType(2, bc_id)
                    if btype in bcs_type[vol]:
                        bcs_type[vol][btype].append(bc_id)
                    else:
                        bcs_type[vol][btype] = [bc_id]
        
        print('Boundary classification:')
        for vol, types in bcs_type.items():
            print(f'  Volume {vol}: {dict(types)}')
        
        return bcs_type


    def create_boundary_groups(self, volume_ids: List[int], bcs: Dict, bcs_type: Dict, 
                            interface: List[int], prefix = "") -> Dict:
        """Create physical groups for boundaries.
        
        Args:
            volume_ids: List of volume IDs
            bcs: Boundary conditions dictionary
            bcs_type: Boundary type classification
            interface: Interface boundary IDs
        """
        bc_names = {}
        eps = self.config.eps
        r1 = self.config.r1
        r2 = self.config.r2
        
        # Get bounding box for z-coordinates
        bbox = gmsh.model.occ.getBoundingBox(3, volume_ids[0])
        zmin, zmax = bbox[2], bbox[5]
        
        # V0 and V1 (top and bottom surfaces)
        V0 = gmsh.model.getEntitiesInBoundingBox(
            -r2-eps, -r2-eps, zmin-eps, 
            r2+eps, r2+eps, zmin+eps, 
            2
        )
        _id =gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in V0], name=f"{prefix}V0")
        print(f'Created V0 group: {len(V0)} surfaces')
        bc_names[f"{prefix}V0"] = _id

        V1 = gmsh.model.getEntitiesInBoundingBox(
            -r2-eps, -r2-eps, zmax-eps, 
            r2+eps, r2+eps, zmax+eps, 
            2
        )
        _id = gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in V1], name=f"{prefix}V1")
        print(f'Created V1 group: {len(V1)} surfaces')
        bc_names[f"{prefix}V1"] = _id

        # Rint (inner radius surfaces)
        rint = gmsh.model.getEntitiesInBoundingBox(
            -r1-eps, -r1-eps, zmin-eps, 
            r1+eps, r1+eps, zmax+eps, 
            2
        )
        
        rint_cu = [tag for (dim, tag) in rint 
                if tag in bcs[volume_ids[0]] and tag in bcs_type[volume_ids[0]]['Cylinder']]
        gmsh.model.addPhysicalGroup(2, rint_cu, name=f"{prefix}Rint")
        print(f'Created Rint group (Cu): {len(rint_cu)} surfaces')
        
        rint_glue = [tag for (dim, tag) in rint 
                    if tag in bcs[volume_ids[1]] and tag in bcs_type[volume_ids[1]]['Cylinder']]
        gmsh.model.addPhysicalGroup(2, rint_glue, name=f"{prefix}iRint")
        print(f'Created iRint group (Glue): {len(rint_glue)} surfaces')
        
        # Rext (outer radius surfaces)
        rext_cu = [tag for tag in bcs[volume_ids[0]] 
                if (2, tag) not in rint and (2, tag) not in V0+V1 and tag not in interface]
        gmsh.model.addPhysicalGroup(2, rext_cu, name=f"{prefix}Rext")
        print(f'Created Rext group (Cu): {len(rext_cu)} surfaces')
        
        rext_glue = [tag for tag in bcs[volume_ids[1]] 
                    if (2, tag) not in rint and (2, tag) not in V0+V1 and tag not in interface]
        gmsh.model.addPhysicalGroup(2, rext_glue, name=f"{prefix}iRext")
        print(f'Created iRext group (Glue): {len(rext_glue)} surfaces')
        
        return bc_names

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
        gmsh.model.add("helix")
        
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration from JSON file
        data = load_config(args.config)
        
        # Parse components
        helixconfig = HelixConfig.from_dict(data)
        helix = Helix(helixconfig, add_start_hole=helixconfig.add_start_hole)

        # Generate geometry
        volume_ids = helix.generate()
        
        # Create physical groups
        helix.create_physical_groups(volume_ids)
        
        # Generate mesh if requested
        if args.mesh:
            helix.generate_mesh(args.mesh_size)
        
        print("\n" + "=" * 60)
        print("Helix generation completed successfully")
        print("=" * 60)

        # Show GUI if not suppressed
        if not args.nopopup:
            gmsh.fltk.run()
            
    except Exception as e:
        print(f"\nError during helix generation: {e}")
        raise
    finally:
        gmsh.finalize()  # Uncomment if you want to finalize Gmsh


if __name__ == "__main__":
    main()
