#!/bin/bash
#SBATCH -J eye_heatfluid        # name of the job
#SBATCH -N 4                    # number of nodes
#SBATCH --ntasks-per-node=128   # number of MPI tasks per node
#SBATCH --threads-per-core=1    # no hyperthreading
#SBATCH -t 48:00:00             # walltime
#SBATCH -p public               # partition
##SBATCH --exclusive            # exclusive mode

#SBATCH -o log/%j-eye_heatfluid-o.log # standard output
#SBATCH -e log/%j-eye_heatfluid-e.log # standard error


export OMP_NUM_THREADS=1
module load hpcx

MESH_INDEX=M5
SOLVER_TYPE=simple
POSITION=prone  # prone supine standing
DISC=P2-P2P1

# ts.cfg
mpiexec -bind-to core feelpp_toolbox_heatfluid \
        --directory=toolboxes/heatfluid/eyeheatfluid/prone/${DISC}-${MESH_INDEX} \
        --case.discretization=${DISC} \
        --config-files eye-${POSITION}.cfg pc_${SOLVER_TYPE}.cfg \
        --heat-fluid.json.patch='{ "op": "replace", "path": "/Meshes/heatfluid/Import/filename", "value": "$cfgdir/mesh/Mr/'${MESH_INDEX}'/Eye_Mesh3D_p$np.json" }' \
        --heat-fluid.scalability-save=1 --heat-fluid.heat.scalability-save=1 --heat-fluid.fluid.scalability-save=1















