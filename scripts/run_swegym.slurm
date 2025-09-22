#!/bin/bash

#SBATCH --account=account_name # TODO: change to the correct account name
#SBATCH --partition=batch # TODO: change to the correct partition
#SBATCH --cpus-per-task=96
#SBATCH --time=23:59:00
#SBATCH --mem=1000G
#SBATCH --gpus-per-node=1
#SBATCH --job-name=mini-swe-agent
#SBATCH --output=mini-swe-agent_%j.out
#SBATCH --error=mini-swe-agent_%j.err

# Set Singularity cache to local /tmp to avoid Lustre issues
# Create cache directories

export SINGULARITY_CACHEDIR=/tmp/singularity_cache_${SLURM_JOB_ID}_$$
export APPTAINER_CACHEDIR=/tmp/apptainer_cache_${SLURM_JOB_ID}_$$
export SINGULARITY_TMPDIR=/tmp/singularity_tmp_${SLURM_JOB_ID}_$$
export APPTAINER_TMPDIR=/tmp/apptainer_tmp_${SLURM_JOB_ID}_$$


mkdir -p $SINGULARITY_CACHEDIR $APPTAINER_CACHEDIR $SINGULARITY_TMPDIR $APPTAINER_TMPDIR

CONTAINER_IMAGE=/lustre/fsw/portfolios/llmservice/users/sdevare/repos/containers/singularity.sqsh # TODO: change to the correct container image
CONTAINER_MOUNTS="/lustre:/lustre,/home:/home,/tmp:/tmp,/lustre/fsw/portfolios/llmservice/users/sdevare/repos/min-swe-agent/:/min-swe-agent" # TODO: change to the correct container mounts
CONTAINER_WORKDIR=/min-swe-agent 

srun --container-image=$CONTAINER_IMAGE \
        --container-mounts=$CONTAINER_MOUNTS \
        --container-workdir=$CONTAINER_WORKDIR \
        /bin/bash -c "
        source $CONTAINER_WORKDIR/.venv/bin/activate

        export SINGULARITY_CACHEDIR=$SINGULARITY_CACHEDIR
        export APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR
        export SINGULARITY_TMPDIR=$SINGULARITY_TMPDIR
        export APPTAINER_TMPDIR=$APPTAINER_TMPDIR
        export TMPDIR="/tmp/"


        # unset SINGULARITY_DOCKER_USERNAME
        # unset SINGULARITY_DOCKER_PASSWORD
        # unset SINGULARITY_TMPDIR

        export OPENAI_API_KEY=???

        mini-extra swegym \
            --subset gym \
            --split train \
            --workers 32 \
            --output $CONTAINER_WORKDIR/outputs \
            --model gpt-4.1-mini \
            --api_key $OPENAI_API_KEY \
            --base_url  https://api.openai.com/v1 \
            --env singularity \
            --step_timeout 600 \
            --eval_timeout 1800 \
            --cache_dir_template /lustre/fsw/portfolios/llmservice/users/sdevare/gym_images/gym/xingyaoww_sweb.eval.x86_64.\{instance_id\}_latest.sif \
        "

# Cleanup temporary cache directories
echo "Cleaning up temporary cache directories..."
rm -rf $SINGULARITY_CACHEDIR $APPTAINER_CACHEDIR $SINGULARITY_TMPDIR $APPTAINER_TMPDIR