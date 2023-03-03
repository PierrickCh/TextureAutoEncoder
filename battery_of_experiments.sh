#bash code to launch experiments with varying parameters
for lam_sp in 1e5 1e7; do 
    for lam_hist in 1 ; do 
        sbatch --gres=gpu:quadro-rtx6000 \
        ./launch_job.sh sp_${lam_sp}_hist_${lam_hist} $lam_hist $lam_sp; 
    done;
done;