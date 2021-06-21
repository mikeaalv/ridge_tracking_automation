#! /bin/bash
#SBATCH --job-name=ridge_seg        # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name
#SBATCH --gres=gpu:P100:1                 # Requests one GPU device
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=20gb                     # Job memory request
#SBATCH --time=18:00:00               # Time limit hrs:min:sec
#SBATCH --output=ridge_seg.%j.out     # Standard output log
#SBATCH --error=ridge_seg.%j.err      # Standard error log
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuewu_mike@163.com  # Where to send mail

cd $SLURM_SUBMIT_DIR

module load Detectron2/0.3-fosscuda-2019b-Python-3.7.4-PyTorch-1.6.0

time python ridge_seg_train.py 1>> ./testmodel.${SLURM_JOB_ID}.out 2>> ./testmodel.${SLURM_JOB_ID}.err
