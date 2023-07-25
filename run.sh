#!/usr/bin/env sh

now=$(date +"%Y%m%d_%H%M%S")

source activate
conda activate tp
srun -p OpenDialogLab_S2  --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=SchemeFree  \
      python transfer_format.py

#     'msg=hello world'
#     &>out_sci.txt




#source /mnt/petrelfs/share/platform/env/pat_diopi_dev_cuda11_torch10.1


