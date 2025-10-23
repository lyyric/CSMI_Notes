#!/bin/bash
#SBATCH -J HL-31_H1             # name of the job
#SBATCH -N 1                    # number of nodes
#SBATCH --ntasks-per-node=4     # number of MPI tasks per node
#SBATCH --threads-per-core=1    # no hyperthreading
#SBATCH -t 48:00:00             # walltime
#SBATCH -p public               # partition
##SBATCH --exclusive            # exclusive mode

#SBATCH -o log/%j-HL-31_H1-o.log # standard output
#SBATCH -e log/%j-HL-31_H1-e.log # standard error


export OMP_NUM_THREADS=1
module load hpcx


# ts.cfg
mpiexec -bind-to core feelpp_toolbox_thermoelectric \
        --config-file HL-31_H1.cfg














