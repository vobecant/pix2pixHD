#!/bin/bash

RUN_NAMES=( run0 run1 run2 run3 run4)
BASE_SAVE_DIR=/home/vobecant/datasets/pix2pixhd/crops

for RN in "${RUN_NAMES[@]}"
do

    EXPNAME="p2p_${RN}"
    SAVE_DIR="${BASE_SAVE_DIR}/${RN}"
    OUT_FILE="${EXPNAME}.out"

    job_file="${EXPNAME}.job"
    echo "run job_file ${job_file}"

    echo "#!/bin/bash
#SBATCH --job-name=${EXPNAME}
#SBATCH --output=${EXPNAME}.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

python -u insert_random.py \
        --name label2city_1024p --netG local --ngf 32 --resize_or_crop none \
        --dataroot /home/vobecant/datasets/cityscapes/ \
        --save_dir ${SAVE_DIR} > ${EXPNAME}.out" > $job_file
    sbatch $job_file

	echo ""

done

