#!/bin/bash
#SBATCH --job-name=slime-rl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1024G
#SBATCH --partition=AISS2025031801
#SBATCH --account polyullm
#SBATCH --time=192:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
# set -ex
# replace these information with your own
workdir=/home/projects/polyullm/
container_image=/lustre/projects/polyullm/container/slime.sqsh
container_name="slimerl-infix-cu129-py312-0914-${SLURM_JOB_ID:-default}"
container_mounts=/lustre/projects/polyullm:/lustre/projects/polyullm,/home/projects/polyullm:/home/projects/polyullm

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

export NVIDIA_DRIVER_CAPABILITIES=all
export NVIDIA_VISIBLE_DEVICES=all

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$workdir \
    --container-writable \
    --container-env=NVIDIA_DRIVER_CAPABILITIES,NVIDIA_VISIBLE_DEVICES \
    bash -c "ray stop && ray start --head --node-ip-address=$head_node_ip --port=$port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &

sleep 5

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        --container-name=$container_name \
        --container-mounts=$container_mounts \
        --container-image=$container_image \
        --container-workdir=$workdir \
        --container-writable \
        --container-env=NVIDIA_DRIVER_CAPABILITIES,NVIDIA_VISIBLE_DEVICES \
        bash -c "ray stop && ray start --address $ip_head --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &
    sleep 5
done

echo "Waiting for 60 seconds..."
sleep 60
echo "Starting RL training..."

SCRIPTS="
set -x
bash $@
"

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$workdir \
    --container-writable \
    --container-env=NVIDIA_DRIVER_CAPABILITIES,NVIDIA_VISIBLE_DEVICES \
    bash -c "$SCRIPTS"


# Clean up Ray processes
cleanup() {
    echo "Shutting down Ray cluster..."
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        --container-name=$container_name \
        --container-mounts=$container_mounts \
        --container-image=$container_image \
        --container-writable \
        --container-env=NVIDIA_DRIVER_CAPABILITIES,NVIDIA_VISIBLE_DEVICES \
        bash -c "ray stop"

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        srun --overlap --nodes=1 --ntasks=1 -w "$node_i" \
            --container-name=$container_name \
            --container-mounts=$container_mounts \
            --container-image=$container_image \
            --container-writable \
            --container-env=NVIDIA_DRIVER_CAPABILITIES,NVIDIA_VISIBLE_DEVICES \
            bash -c "ray stop"
    done
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT



