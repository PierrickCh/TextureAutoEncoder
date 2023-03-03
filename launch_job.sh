#!/bin/bash
#SBATCH --job-name=ffc_TAE   # -J nom-job      => nom du job
#SBATCH --ntasks=4        # -n 4            => nombre de taches (obligatoire)
#SBATCH --time 7-0:00          # -t 0-2:00       => duree (D-HH:MM) (obligatoire)
#SBATCH --qos=co_long_gpu 
#SBATCH --output=/scratchm/pchatill/slurm/out/slurm.%j.out  # -o slurm.%j.out => Sortie standard
#SBATCH --error=/scratchm/pchatill/slurm/err/slurm.%j.err  # -e slurm.%j.err => Sortie Erreur


source ~/.bashrc
conda activate TAE
cd /scratchm/pchatill/projects/git_clean/TextureAutoEncoder
python3  code/train.py --dirname $1 --lam_hist $2 --lam_sp $3  -n_iter 10000
