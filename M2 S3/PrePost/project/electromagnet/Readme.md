# Insert Geometry Generator

A modular Python framework for generating electromagnets consisting of helices and rings using Gmsh.

## Installation

```bash
# Create a virtual python env
python -m venv .venv
source .venv/bin/activate

# Install Gmsh
pip install gmsh

# Run the script
python insert.py -config your_config.json

# Exit from virtual env
deactivate
```

## Usage

### Basic Usage

```bash
python insert.py -config insert_config.json
```

### With Mesh Generation

```bash
python insert.py -config insert_config.json -mesh
```

### Custom Mesh Size

```bash
python insert.py -config insert_config.json -mesh -mesh-size 2.0
```

### Without GUI

```bash
python insert.py -config insert_config.json -mesh -nopopup
```

## JSON Configuration Format

### Complete Example

```json
{
  "helices": [
    {
      "name": "helix_bottom",
      "r1": 19.3,
      "r2": 24.2,
      "dz": 300,
      "cut": 0.2,
      "eps": 2,
      "nturns": 1,
      "pitch": 18.03646917296748,
      "npts": 60,
      "start_hole_diameter": 1.6,
      "z_offset": 0.0,
      "add_start_hole": true
    }
  ],
  "rings": [
    {
      "name": "ring_middle",
      "r1": 19.3,
      "r2": 30.7,
      "h": 20,
      "n": 4,
      "r_slit": 24.65,
      "e_slit": 0.9,
      "angular_length": 30,
      "z_offset": 300.0,
      "add_fillet": true
    }
  ]
}
```

### Helix Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Component identifier |
| `r1` | float | Yes | Inner radius |
| `r2` | float | Yes | Outer radius |
| `dz` | float | Yes | Cylinder height |
| `cut` | float | Yes | Cut width |
| `eps` | float | Yes | Tolerance/epsilon |
| `nturns` | int | Yes | Number of helix turns |
| `pitch` | float | Yes | Helix pitch |
| `npts` | int | No | Number of points along helix (default: 60) |
| `start_hole_diameter` | float | No | EDM wire hole diameter (default: 1.6) |
| `z_offset` | float | No | Vertical position offset (default: 0.0) |
| `add_start_hole` | bool | No | Add EDM wire start hole (default: false) |

### Ring Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Component identifier |
| `r1` | float | Yes | Inner radius |
| `r2` | float | Yes | Outer radius |
| `h` | float | Yes | Height |
| `n` | int | Yes | Number of slits |
| `r_slit` | float | Yes | Radius of slit |
| `e_slit` | float | Yes | Radial width of slit |
| `angular_length` | float | Yes | Angular length of slit (degrees) |
| `z_offset` | float | No | Vertical position offset (default: 0.0) |
| `add_fillet` | bool | No | Add fillets to slits (default: false) |

## Features

### Component Positioning

Use `z_offset` to position components vertically:
- Stack multiple helices at different heights
- Position rings between helices
- Create complex multi-level geometries

### Physical Groups

The Insert automatically creates named physical groups:
- **Helix components**: `{name}_Cu` and `{name}_Glue`
- **Ring components**: `{name}_Ring`

Example for a helix named "helix_bottom":
- Physical group: `helix_bottom_Cu`
- Physical group: `helix_bottom_Glue`

## Examples

### Single Helix

```json
{
  "helices": [
    {
      "name": "helix_main",
      "r1": 19.3,
      "r2": 24.2,
      "dz": 300,
      "cut": 0.2,
      "eps": 2,
      "nturns": 1,
      "pitch": 18.03646917296748,
      "add_start_hole": true
    }
  ],
  "rings": []
}
```

### Single Ring

```json
{
  "helices": [],
  "rings": [
    {
      "name": "ring_main",
      "r1": 19.3,
      "r2": 30.7,
      "h": 20,
      "n": 4,
      "r_slit": 24.65,
      "e_slit": 0.9,
      "angular_length": 30,
      "add_fillet": true
    }
  ]
}
```

### Stacked Assembly

```json
{
  "helices": [
    {
      "name": "helix_bottom",
      "r1": 19.3,
      "r2": 24.2,
      "dz": 200,
      "cut": 0.2,
      "eps": 2,
      "nturns": 1,
      "pitch": 18.03646917296748,
      "z_offset": 0.0
    },
    {
      "name": "helix_top",
      "r1": 19.3,
      "r2": 24.2,
      "dz": 200,
      "cut": 0.2,
      "eps": 2,
      "nturns": 1,
      "pitch": 18.03646917296748,
      "z_offset": 250.0
    }
  ],
  "rings": [
    {
      "name": "ring_spacer",
      "r1": 19.3,
      "r2": 30.7,
      "h": 30,
      "n": 4,
      "r_slit": 24.65,
      "e_slit": 0.9,
      "angular_length": 30,
      "z_offset": 200.0
    }
  ]
}
```

## Architecture


### Core Classes

#### `Insert`
Main class that manages a collection of helix and ring components.
- Reads JSON configuration
- Creates and manages multiple `Helix` and `Ring` objects
- Coordinates geometry generation and meshing
- Creates physical groups for all components

#### `Helix`
Represents a helical geometry component.
- Generates helix geometry with optional EDM wire start hole
- Creates Cu and Glue volumes
- Supports vertical positioning via `z_offset`

#### `Ring`
Represents a ring geometry with slits.
- Generates ring geometry with configurable number of slits
- Optional fillet features
- Supports vertical positioning via `z_offset`

#### Configuration Classes
- `HelixConfig`: Data class for helix parameters
- `RingConfig`: Data class for ring parameters

## API Reference

### Insert Class

#### Methods

```python
__init__(config_path: str)
```
Initialize insert from JSON configuration file.

```python
parse_components() -> None
```
Parse the JSON configuration and create Helix and Ring objects.

```python
generate_geometry() -> None
```
Generate geometry for all components.

```python
create_physical_groups() -> None
```
Create physical groups for all components.

```python
generate_mesh(mesh_size: Optional[float] = None) -> None
```
Generate mesh for the entire insert.

### Helix Class

#### Methods

```python
__init__(config: HelixConfig, add_start_hole: bool = False)
```
Initialize helix component.

```python
generate() -> List[int]
```
Generate the helix geometry. Returns list of volume IDs.

```python
create_physical_groups() -> None
```
Create physical groups for the helix (Cu and Glue volumes).

### Ring Class

#### Methods

```python
__init__(config: RingConfig)
```
Initialize ring component.

```python
generate() -> List[int]
```
Generate the ring geometry. Returns list of volume IDs.

```python
create_physical_groups() -> None
```
Create physical groups for the ring.

## Output Files

### Mesh File
- Default: `insert.msh`
- Gmsh mesh format containing all components

### Physical Groups
Physical groups are automatically created with naming convention:
- `{component_name}_Cu` - Copper volume (helix only)
- `{component_name}_Glue` - Glue volume (helix only)
- `{component_name}_Ring` - Ring volume

## Error Handling

The framework includes comprehensive error handling:
- **File not found**: Clear message if config file missing
- **Invalid JSON**: Reports JSON parsing errors
- **Missing parameters**: Identifies missing required parameters
- **Geometry errors**: Catches and reports Gmsh errors


### Mesh Size
Start with automatic mesh size, then refine:
```bash
# Auto mesh
python insert.py -config config.json -mesh

# Custom mesh for finer control
python insert.py -config config.json -mesh -mesh-size 0.5
```

### 4. Configuration Organization
Keep separate config files for different designs:
```
configs/
  ├── insert_prototype_v1.json
  ├── insert_prototype_v2.json
  ├── insert_production.json
  └── insert_test_simple.json
```

## Troubleshooting

### Issue: Components not visible in GUI
- Check `z_offset` values aren't overlapping
- Verify radius parameters (r1 < r2)
- Use `-nopopup` flag and check console output

### Issue: Mesh generation fails
- Reduce `-mesh-size` for finer mesh
- Check for geometry intersections
- Verify component parameters are physically valid

### Issue: Missing physical groups
- Ensure component names are unique
- Check that `generate_geometry()` completed successfully
- Verify volume IDs were created

## Requirements

- Python 3.7+
- Gmsh Python API
- Standard library: `json`, `argparse`, `dataclasses`, `typing`


