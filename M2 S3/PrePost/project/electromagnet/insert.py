"""
Insert geometry manager using Gmsh.

This module manages complex geometries composed of multiple helix and ring components.
The Insert class reads a JSON configuration and creates all components.
"""

import gmsh
import math
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from helix import Helix, HelixConfig
from ring import Ring, RingConfig

def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a nested list into a single list."""
    return [item for sublist in nested_list for item in sublist]

def load_config(config_path) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded insert configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        raise

class Insert:
    """Insert composed of multiple helix and ring components."""
    
    def __init__(self, config_path: str):
        """Initialize insert from JSON configuration.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self.helices: List[Helix] = []
        self.rings: List[Ring] = []
        self.hangles: List[float] = []
        self.rangles: List[float] = []
        self.config = load_config(config_path)
        
    
    def parse_components(self):
        """Parse component configurations and create component objects."""
        print("\n=== Parsing Insert Components ===")
        
        # Parse helices
        if 'helices' in self.config:
            for helix_data in self.config['helices']:
                helix_config = HelixConfig.from_dict(helix_data)
                add_start_hole = helix_data.get('add_start_hole', False)
                helix = Helix(helix_config, add_start_hole)
                self.helices.append(helix)
                print(f"  Added helix: {helix_config.name}")
        
        # Parse rings
        if 'rings' in self.config:
            for ring_data in self.config['rings']:
                ring_config = RingConfig.from_dict(ring_data)
                ring = Ring(ring_config)
                self.rings.append(ring)
                print(f"  Added ring: {ring_config.name}")
        
        self.hangles = self.config.get('hangles', [])
        self.rangles = self.config.get('rangles', [])

        print(f"\nTotal components: {len(self.helices)} helices, {len(self.rings)} rings")
    
    def generate_geometry(self):
        """Generate geometry for all components."""
        print("\n" + "=" * 60)
        print("Generating Insert Geometry")
        print("=" * 60)
        
        # Generate all helices
        helices_ids = []
        ignore_ids = []
        for i, helix in enumerate(self.helices):
            helix_ids = helix.generate(ignore_ids)
            ignore_ids.extend(helix_ids)
            # print(f'H[{i+1}]: {helix_ids}')
            ################ TO FILL IN THE ELLIPSES WITH PROPER ARGUMENTS ################
            if self.hangles and i < len(self.hangles) and self.hangles[i] != 0:
                angle = math.radians(self.hangles[i])
                gmsh.model.occ.rotate(
                    [(3, vid) for vid in helix_ids],
                    0, 0, 0,
                    0, 0, 1,
                    angle
                )
                gmsh.model.occ.synchronize()

            helices_ids.append(helix_ids)
        # print(f'helices_ids: {helices_ids}')

        # Generate all rings
        rings_ids = []
        for i, ring in enumerate(self.rings):
            ring_id = ring.generate()
            # print(f'R[{i+1}]: {ring_id}')
            
            # position ring on z (ring(i): Helix(i) to Helix(i+1) )
            h = self.helices[i].config.nturns * self.helices[i].config.pitch
            _z = 0
            if i%2 == 0:
                _z = self.helices[i].config.z2
            else:
                _z = -(self.helices[i].config.z1 + self.ring[i].config.h)
            
            _z += self.helices[i].config.z_offset

            print(f'  Translating ring R[{i+1}] to z={_z}')
            gmsh.model.occ.translate([(3, rid) for rid in ring_id], 0, 0, _z)
            gmsh.model.occ.synchronize()
            
            ################ TO FILL IN THE ELLIPSES WITH PROPER ARGUMENTS ################
            if self.rangles and i < len(self.rangles) and self.rangles[i] != 0:
                angle = math.radians(self.rangles[i])
                gmsh.model.occ.rotate(
                    [(3, rid) for rid in ring_id],
                    0, 0, 0,
                    0, 0, 1,
                    angle
                )
                gmsh.model.occ.synchronize()
            
            rings_ids.append(ring_id)
        # print(f'rings_ids: {rings_ids}')
        
        # assembly
        print('\n=== Assembling Helices and Rings ===')
        helices_dimtags = [(3, id) for id in flatten_list(helices_ids)]
        rings_dimtags = [(3, id) for id in flatten_list(rings_ids)]
        
        # print(f'helices_dimtags: {helices_dimtags},  rings_dimtags: {rings_dimtags}')
        ################ TO FILL IN THE ELLIPSES WITH PROPER ARGUMENTS ################
        all_dimtags = [(3, id) for id in flatten_list(helices_ids + rings_ids)]
        outDimTags, outDimTagsMap = gmsh.model.occ.fragment(all_dimtags, all_dimtags)
        gmsh.model.occ.synchronize()
        
        # Display parent-child relationships
        print("Parent-child fragment relations:")
        children_dict = {}
        for parent, children in zip(helices_dimtags + rings_dimtags, outDimTagsMap):
            # print(f"  Parent {parent} -> Children {children}")
            for child in children:
                if parent[1] not in children_dict:
                    children_dict[parent[1]] = [child[1]]
                else:
                    children_dict[parent[1]].append(child[1])
        print(f'children_dict: {children_dict}')

        cyl_children_id = [dimtag[1] for dimtag in outDimTagsMap[0]]
        # get volume ids - see fragment_geometry in helix_restructured
        # print('done')

        for j, helix_id in enumerate(helices_ids):
            old_ids = helix_id.copy()
            helices_ids[j] = flatten_list([children_dict[id] for id in old_ids])
            print(f'helix_id old: {old_ids} --> new: {helices_ids[j]}')
            for i, id in enumerate(old_ids):
                if id != helices_ids[j][i]:
                    print('remove helix volume id:', id)
                    gmsh.model.occ.remove([(3, id)], False)
        print(f'helices_ids: {helices_ids}')
        
        for j, ring_id in enumerate(rings_ids):
            old_ids = ring_id.copy()
            rings_ids[j] = flatten_list([children_dict[id] for id in old_ids])
            print(f'ring_id old: {old_ids} --> new: {rings_ids[j]}')
            for i, id in enumerate(old_ids):
                if id != rings_ids[j][i]:
                    print('remove ring volume id:', id)
                    gmsh.model.occ.remove([(3, id)], False)
        print(f'rings_ids: {rings_ids}')

        gmsh.model.occ.synchronize()

        return helices_ids, rings_ids #, children_dict

    def create_physical_groups(self, helices_ids: List[List[int]], rings_ids: List[List[int]]): #, children_dict   ):
        """Create physical groups for all components."""
        print("\n=== Creating Physical Groups for Insert ===")

        bcs_names = {}
        for i,helix in enumerate(self.helices):
            _names = helix.create_physical_groups(helices_ids[i], helix.config.name)
            bcs_names.update(_names)

        # TODO: rename physical for ring: V1 for odd ring --> BP, V0 for even ring --> HP
        for i, ring in enumerate(self.rings):
            ring.create_physical_groups(rings_ids[i], ring.config.name)
        
        # need to drop physical for V1 for 1st helix and last helix , V0 and V1 for the others
        gmsh.model.removePhysicalGroups([(2, bcs_names["H1_V1"])])
        gmsh.model.removePhysicalGroups([(2, bcs_names[f"H{len(self.helices)}_V1"])])
        for i in range(2, len(self.helices)):
            gmsh.model.removePhysicalGroups([(2, bcs_names[f"H{i}_V0"])])
            gmsh.model.removePhysicalGroups([(2, bcs_names[f"H{i}_V1"])])

        # TODO: group BCs for channels between helices 
    
    def generate_mesh(self, helices_ids: List[List[int]], rings_ids: List[List[int]]):
        """Generate mesh for the entire insert.
        
        Args:
            mesh_size: Global mesh size (if None, auto-calculated)
        """
        print("\n=== Generating Mesh ===")

        # Set mesh options
        gmsh.option.setNumber('Geometry.NumSubEdges', 1000)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 40)
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.1)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 1)
        gmsh.option.setNumber('Mesh.AngleToleranceFacetOverlap', 0.1)

        ################ IMPLEMENT MESH_SIZE per components ################
        """
        add argument mesh_size to the method:

        for i, helix in enumerate(self.helices):
            mesh_size = abs(helix.config.r2 - helix.config.r1) / 3.
            use gmsh.model.getBoundary to get points of helix volumes (see helices_ids)
            use gmsh.model.mesh.setSize to set mesh_size to the points

        same for ring
        for i, ring in enumerate(self.rings):
            mesh_size = abs(ring.config.r2 - ring.config.r1) / 3.
        
        do the order of setting mesh sizes matter ? 
        should we consider to re-order the mes size defs ?
        how can this be done properly ?
        """
        ################################


        # Generate 3D mesh
        print('Generating 3D mesh...')
        OUTPUT_MESH_FILE = self.config["name"] + ".msh"
        gmsh.model.mesh.generate(3)
        gmsh.write(OUTPUT_MESH_FILE)
        print(f'Mesh written to: {OUTPUT_MESH_FILE}')


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Generate insert geometry with multiple helix and ring components',
        add_help=True
    )
    parser.add_argument(
        '-config', 
        type=str, 
        required=True,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '-mesh', 
        action='store_true',
        help='Generate mesh after geometry creation'
    )
    parser.add_argument(
        '-nopopup', 
        action='store_true',
        help='Do not show Gmsh GUI'
    )
    parser.add_argument(
        '-mesh-size',
        type=float,
        default=None,
        help='Global mesh size (optional)'
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")
    return args


def main():
    """Main function to orchestrate the insert generation process."""
    print("=" * 60)
    print("Insert Geometry Generator")
    print("=" * 60)
    
    #try:
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("insert")
    
    # Parse command line arguments
    args = parse_arguments()
    print('args:', args)
    
    # Create insert and load configuration
    insert = Insert(args.config)
    
    # Parse components
    insert.parse_components()
    
    # Generate geometry
    # helices_ids, rings_ids, children_dict = insert.generate_geometry()
    helices_ids, rings_ids = insert.generate_geometry()
    
    # Create physical groups
    insert.create_physical_groups(helices_ids, rings_ids) #, children_dict)
    
    # Generate mesh if requested
    if args.mesh:
        # insert.generate_mesh(helices_ids, rings_ids, children_dict, args.mesh_size)
        insert.generate_mesh(helices_ids, rings_ids)
    
    print("\n" + "=" * 60)
    print("Insert generation completed successfully")
    print("=" * 60)
    
    # Show GUI if not suppressed
    if not args.nopopup:
        gmsh.fltk.run()
            
    """
    except Exception as e:
        print(f"\nError during insert generation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        gmsh.finalize()  # Uncomment if you want to finalize Gmsh
    """

if __name__ == "__main__":
    main()
