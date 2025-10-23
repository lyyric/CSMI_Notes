# Electromagnet Thermoelectric Modeling Project

## Project Overview

This project focuses on studying an electromagnet by modeling its thermoelectric behavior using the Feel++ Thermoelectric toolbox. The electromagnet consists of two helices and one connection ring. All calculations will be performed on the Gaya computing cluster.

## Project Structure

The project is divided into three main parts:

1. **CAD Geometry and Mesh Creation** - Using Gmsh Python API
2. **Problem Setup and Solution** - Using Feel++ Thermoelectric toolbox
3. **Post-processing Operations** - Analysis and visualization of results

---

## Part 1: CAD Geometry and Mesh Generation

### Prerequisites
- Retrieve the package from Moodle
- Access to Gmsh Python API
- Python environment configured

### Step 1: Package Setup

**Tasks:**
- Download the package from Moodle
- Untar the package using: `tar -xvf package_name.tar.gz`
- Read the Readme.md file carefully
- Follow the installation instructions provided in the README

### Step 2: Initial Testing - Connection Ring

**Tasks:**
- Complete create_boundary_groups in ring.py (see ## TO FILL IN THE ELLIPSES WITH PROPER ARGUMENTS ## )
- Test the CAD generation for a connection ring
- Verify that the geometry is generated correctly (compare with attached ring.geo)
- Document any issues encountered

### Step 3: Electromagnet (Insert) Generation

**Tasks:**
- Complete the package to generate the full electromagnet geometry (see  ## TO FILL IN THE ELLIPSES WITH PROPER ARGUMENTS ## in helix.py, insert.py)
- Implement the generation of both helices
- Ensure proper integration with the connection ring

### Step 4: Extra Features Implementation

Implement the following additional features:

**rangles Parameter:**
- Implement the rangle (orientation of connection ring within insert) feature
- Document its purpose and usage

**hangle Parameter:**
- Implement the hangle (orientation of helix within insert) feature
- Document its purpose and usage

### Step 5: Mesh Generation - Ring

**Tasks:**
- Test the mesh generation process for the connection ring
- Verify mesh quality and element distribution

### Step 6: Mesh Generation - Complete Insert

**Tasks:**
- Complete the mesh generation for the full electromagnet insert
- Ensure proper mesh continuity between components
- Add selection of 2D and 3D mesh algo from the command line
  - Add args to parse_arguments method in insert.py (use `choice` argparse)
  - Change the required method to use the 2D, 3D selected algo
  
### Step 7: Advanced Mesh Features

Implement the following mesh enhancements:

**Different Mesh Sizes:**
- Configure distinct mesh sizes for helices
- Configure distinct mesh sizes for rings

**Local Mesh Refinement:**
- Identify critical regions requiring finer mesh
- Implement local refinement strategies

---

## Part 2: Problem Setup and Solution

### Prerequisites
- Completed CAD and mesh generation
- Access to Gaya computing cluster
- Feel++ Thermoelectric toolbox configured

### Step 1: Setup Files

**Tasks:**
- Retrieve the setup files from Moodle for 1 helix
- Review configuration file for 1 helix
- Understand boundary conditions and material properties
- Create setup and configuration files adapted to the insert created in Part1
- Partition the mesh for the simulation on gaya
- Update the setup files to use the partitionned mesh

### Step 2: Data Transfer

**Tasks:**
- Transfer mesh files to Gaya cluster
- Transfer setup/configuration files to Gaya
- Verify file integrity after transfer
- Organize files in appropriate directories

### Step 3: Job Execution

**Tasks:**
- Locate the provided SLURM script
- Review script parameters (number of cores, memory allocation, time limits)
- Adapt the script for insert model
- Check that the number of partitions within the mesh matches the requested number of cores
- Submit the job using: `sbatch script_name.slurm`
- Monitor job status using: `squeue -u your_username`
- Check output logs for errors or warnings

---

## Part 3: Post-processing

### Objectives
- Analyze simulation results
- Visualize temperature and electric field distributions
- Extract relevant physical quantities
- Generate plots and figures for reporting

### Tasks
*[To be completed based on specific requirements]*

- Extract temperature fields
- Visualize current density distributions
- Analyze heat dissipation patterns
- Generate comparison plots
- Document findings

---

## Deliverables Checklist

- [ ] Working CAD generation code (eventually with extra features)
- [ ] Mesh generation scripts for complete electromagnet
- [ ] Successful simulation run on Gaya
- [ ] Post-processing results and visualizations
- [ ] Final report with analysis and conclusions
- [ ] Comment on possible improvements (geometry, mesh, ...)

---

## Important Notes

- Always backup your work before major modifications
- Document all parameter choices and their justifications
- Keep track of simulation parameters for reproducibility
- Save intermediate results at each stage
- Test on simple cases before running full simulations

## Resources

- Moodle: Project package and setup files
- Gaya cluster: Computation platform
- Feel++ Documentation: Thermoelectric toolbox reference
- Gmsh Python API: Geometry and mesh generation reference