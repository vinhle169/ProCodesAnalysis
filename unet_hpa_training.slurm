#!/bin/bash
#SBATCH -J unet_hpa
#SBATCH -o unet_hpa_%j.out
#SBATCH -e unet_hpa_%j.err
#SBATCH --mail-user=vinhle@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=1000GB
#SBATCH --time=24:00:00
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=procodes
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

## Horovod execution
horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python /home/vinhle/procodes/code/unet_lightning.py 1000 '/nobackup/users/vinhle/data/hpa_data/hpa_train/train/' '/nobackup/users/vinhle/data/hpa_data/hpa_train/truth/' '/home/vinhle/procodes/code/models/unet/' --batch_size 64 --gpus 2

echo "Run completed at:- "
date