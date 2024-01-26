#!/bin/bash
export PYTHONUNBUFFERED=1
export OMPI_MCA_mpi_yield_when_idle=0

module load 2022 OpenMPI/4.1.4-GCC-11.3.0
module load SDL2/2.0.22-GCCcore-11.3.0

if grep -q "tcn" <<< `hostname`; then
  source /projects/0/hpmlprjs/PokemonRedExperiments/venv_cpu/bin/activate
else
  source /projects/0/hpmlprjs/PokemonRedExperiments/venv/bin/activate
fi

export PYSDL2_DLL_PATH=$EBROOTSDL2
python run.py

