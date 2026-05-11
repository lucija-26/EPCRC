#!/bin/bash
# ============================================================
#  Submit with:   bsub < experiments/run_sweep_all.sh
# ============================================================

#BSUB -J epcrc_sweep
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 08:00
#BSUB -o sweep_%J.out
#BSUB -e sweep_%J.err

# --- Go to project root ---
cd "$HOME/EPCRC" || { echo "ERROR: ~/EPCRC not found"; exit 1; }

# --- Load Python ---
module load python3

# --- Set up HPC virtual environment (created once, reused on reruns) ---
VENV="$HOME/EPCRC/.venv_hpc"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[setup] Creating HPC venv..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
else
    source "$VENV/bin/activate"
    if ! python -c "import scipy" 2>/dev/null; then
        pip install --quiet -r requirements.txt
    fi
fi

# --- Let numpy/scipy use all allocated cores ---
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export MKL_NUM_THREADS=$LSB_DJOB_NUMPROC
export OPENBLAS_NUM_THREADS=$LSB_DJOB_NUMPROC

# --- Info ---
echo "=============================="
echo "Job    : $LSB_JOBID"
echo "Node   : $(hostname)"
echo "CPUs   : $LSB_DJOB_NUMPROC"
echo "Python : $(python --version)"
echo "Dir    : $(pwd)"
echo "Start  : $(date)"
echo "=============================="

# --- Run ---
python experiments/experiment_0_gamma_sweep_all.py

echo "=============================="
echo "Done   : $(date)"
echo "=============================="
