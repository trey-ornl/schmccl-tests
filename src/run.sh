#!/bin/bash
module load rocm
set -x
export MPICH_GPU_SUPPORT_ENABLED=1
NODES=${1}
TASKS=${2}
shift ; shift
srun -u -t 1:00 --reservation=hack4 -N${NODES} -n${TASKS} -c8 --gpus-per-task=1 --gpu-bind=closest --exclusive ./all_reduce_perf $@
export MPICH_GPU_ALLREDUCE_USE_KERNEL=1
srun -u -t 1:00 --reservation=hack4 -N${NODES} -n${TASKS} -c8 --gpus-per-task=1 --gpu-bind=closest --exclusive ./all_reduce_perf $@
