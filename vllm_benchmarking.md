# vLLM Benchmarking Guide

This guide provides detailed instructions for setting up and running benchmarks using vLLM on Intel Xeon processors. The benchmarking process helps evaluate the performance of large language models (LLMs) in serving scenarios.

## Overview

vLLM is a high-performance framework for LLM inference and serving. This guide covers:

1. Setting up your server environment
2. Configuring Docker
3. Building vLLM for CPUs
4. Running benchmarks with different parallelism strategies:
   - Load Balancing across NUMA nodes
   - Tensor Parallelism

## Prerequisites

- Intel Xeon server (preferably with multiple NUMA nodes)
- Ubuntu OS (tested on Ubuntu 24.04 LTS)
- Python 3.x
- Docker

## Server Setup

### 1. Gathering Server Details

First, examine your server's hardware configuration:
```bash
lscpu
```

Pay special attention to:
- CPU architecture
- Number of cores
- NUMA node configuration
- Thread(s) per core

For example:
```bash
NUMA:
NUMA node(s): 6
NUMA node0 CPU(s): 0-23,144-167
NUMA node1 CPU(s): 24-47,168-191
NUMA node2 CPU(s): 48-71,192-215
NUMA node3 CPU(s): 72-95,216-239
NUMA node4 CPU(s): 96-119,240-263
NUMA node5 CPU(s): 120-143,264-287
```

### 2. Docker Setup

Install Docker following these steps:

```bash
# Uninstall any existing Docker packages
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Set up proxies if needed
export http_proxy=http://proxy.example:123/
export https_proxy=http://proxy.example:123/
export no_proxy=localhost,127.0.0.1,0.0.0.0

# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker packages
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
sudo docker run hello-world

# Add your user to the docker group
sudo usermod -aG docker $USER
```

Log out and log back in to apply the group changes.

#### Docker Proxy Configuration (if needed)

If you're behind a corporate network:

```bash
# Switch to root user
sudo su

# Create systemd directory
mkdir /etc/systemd/system/docker.service.d

# Add proxies
cat <<EOT >> /etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=http://proxy.example:123/"
Environment="HTTPS_PROXY=http://proxy.example:123/"
Environment="NO_PROXY=localhost,127.0.0.0"
EOT

# Reload daemon
systemctl daemon-reload

# Restart docker
systemctl restart docker
```

## Building vLLM for CPU

### 1. Clone the Repository

```bash
git clone https://github.com/vllm-project/vllm
```

If using a corporate proxy:

```bash
export http_proxy=http://proxy.example:123/
export https_proxy=http://proxy.example:123/
export no_proxy=localhost,127.0.0.1,0.0.0.0
```

### 2. Build the Docker Image

```bash
cd vllm
docker build --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile.cpu -t vllm-cpu-env .
```

### 3. Set Up Python Environment

```bash
# Create a new environment
python3 -m venv ~/vllm-bench

# Activate the environment
source ~/vllm-bench/bin/activate
```

### 4. Download Model

```bash
# Install Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Set custom cache directory (optional)
export HF_CACHE=/path/to/model

# Set your Hugging Face token (if using gated models)
export HF_TOKEN=<your-token>

# Download the model
huggingface-cli download <model-name>
```

### 5. Install LLMPerf

```bash
git clone https://github.com/ray-project/llmperf
cd llmperf
pip install -e .
```

Note: If using Python 3.12, you may need to edit the `pyproject.toml` file to adjust requirements.

## Benchmarking Methods

vLLM can be benchmarked using two different parallelism strategies:

### Method 1: Load Balancing Across NUMA Nodes

This method deploys multiple vLLM instances (one per NUMA node) with an NGINX load balancer.

#### Step 1: Generate Configuration Files

First, download the configuration generator script:

```bash
wget https://raw.githubusercontent.com/akarX23/intel-scripts/refs/heads/master/LLMs/vllm-cpu-lb-bench/gen-deploy-files.py
```

View available options:

```bash
python3 gen-deploy-files.py --help
```

Generate the configuration files:

```bash
python3 gen-deploy-files.py \
--core_ranges 0-23,24-47,48-71,72-95,96-119,120-143 \
--model meta-llama/Llama-3.1-8B-Instruct \
--docker_image vllm-cpu-env \
--nginx_port 8000 \
--nginx_core 225-239 \
--kv_cache 80 \
--hf_cache /path/to/hf_cache
```

This will generate `docker-compose.yml` and `nginx.conf` files.

#### Step 2: Run the Benchmark

Install numactl:

```bash
sudo apt install -y numactl
```

Download the benchmark script:

```bash
wget https://raw.githubusercontent.com/akarX23/intel-scripts/refs/heads/master/LLMs/vllm-cpu-lb-bench/bench.sh
chmod +x bench.sh
```

View available options:

```bash
./bench.sh --help
```

Run the benchmark:

```bash
export HF_TOKEN=<your-token>

numactl -C 240-245 ./bench.sh \
--host localhost \
--port 8000 \
--deployment-files-root /path/to/deploy \
# default: /home/cefls_user/intel-scripts/LLMs/vllm-cpu-lb-bench
--dataset-name random \
# default: random
--num-prompts 300 \
# default: 1000
--model meta-llama/Llama-3.1-8B-Instruct \
--concurrencies 128,256 \
--input-lengths 128,256 \
--output-lengths 512,1024 \
--log-dir run1 \
--num-deployments 6 \
--cores-per-deployment 24 \
--llmperf-root /path/to/llmperf
```

Results will be saved in the specified log directory.
To automate a long-running benchmark, the script accepts these arguments for the benchmark client:
```bash
  --concurrencies <list>        List of concurrency values (required, comma-separated)
  --input-lengths <list>        List of input token lengths (required, comma-separated)
  --output-lengths <list>       List of output token lengths (required, comma-separated)
  ```
The script will run every iteration possible of the list of numbers provided in these parameters, one after the other, and compile the results. For example, --concurrencies 1,2 --input-lengths 128,256 --output-lengths 512,1024 will run these combinations:

```bash
Results:
Concurrency,Input Length,Output Length 
1,128,512 
1,128,1024 
1,256,512 
1,256,512 
2,128,512 
2,128,1024 
2,256,512 
2,256,512
```
Lastly, the `--log-dir` parameter will accept a directory to save all logs to, which will be 3 files: 
-	`vllm-server.out` : Output of the vLLM Server command 
-	`client.out`: Output of the benchmark client 
-	`results.csv` : CSV formatted results for all the combinations specified

To record results, you need to specify `--num-deployments` and `--cores-per-deployment`. 

Also, `--llmperf-root` is the path to the LLMPerf directory. 
The --deployment-files-root is the directory where the generated docker-compose.yml and nginx.conf files live.

Additionally, make sure to run the script on a separate set of threads, other than the ones allotted for the vLLM containers and NGINX using numactl


### Method 2: Tensor Parallelism

This method splits the model across NUMA nodes using tensor parallelism.

#### Step 1: Run the vLLM Container

```bash
docker run -d \
-v /home/$USER/.cache/huggingface/hub:/root/.cache/huggingface/hub \
-e http_proxy=$http_proxy \
-e HTTP_PROXY=$http_proxy \
-e https_proxy=$https_proxy \
-e HTTPS_PROXY=$https_proxy \
-e no_proxy=$no_proxy \
-e NO_PROXY=$no_proxy \
--name vllm \
--privileged \
--entrypoint sleep \
vllm-cpu-env infinity
```

#### Step 2: Access the Container Shell

```bash
docker exec -it vllm bash
```

#### Step 3: Download and Run the Benchmark Script

```bash
wget https://raw.githubusercontent.com/akarX23/intel-scripts/refs/heads/master/LLMs/vllm-master-bench.sh
chmod +x vllm-master-bench.sh
```

View available options:

```bash
./vllm-master-bench.sh --help
```

Run the benchmark:

```bash
export HF_TOKEN=<your-token>

./vllm-master-bench.sh \
--cpus_bind "0-23|24-47|48-71|72-95|96-119|120-143" \
--tp 6 \
--model meta-llama/Llama-3.1-8B-Instruct \
--concurrencies 1,2,4,8,16,32,64 \
--input-lengths 128,256,512,1024 \
--output-lengths 128,256,512,1024 \
--log-dir run1
```

## Using benchmark_serving.py directly

For more fine-grained control, you can use the `benchmark_serving.py` script directly from the vLLM repository.

### Key Parameters

- `--backend`: Serving backend (default: "vllm")
- `--host` and `--port`: Server address
- `--model`: Model name/path
- `--dataset-name`: Dataset for benchmarking
- `--num-prompts`: Number of prompts to process
- `--request-rate`: Number of requests per second
- `--concurrencies`: Concurrency levels for testing
- `--input-lengths` and `--output-lengths`: Token lengths
- `--save-result`: Save benchmark results to JSON

### Example Usage

```bash
python benchmarks/benchmark_serving.py \
--backend vllm \
--host localhost \
--port 8000 \
--model meta-llama/Llama-3.1-8B-Instruct \
--dataset-name random \
--num-prompts 300 \
--request-rate 10 \
--max-concurrency 64 \
--random-input-len 256 \
--random-output-len 128 \
--save-result \
--result-dir ./benchmark_results
```

## Benchmark Metrics

The benchmarking process measures several key metrics:

- **Request throughput**: Requests processed per second
- **Output token throughput**: Output tokens generated per second
- **Total token throughput**: Total tokens (input + output) processed per second
- **TTFT (Time to First Token)**: Latency until the first token is generated
- **TPOT (Time per Output Token)**: Average time to generate each token after the first
- **ITL (Inter-token Latency)**: Time between consecutive tokens
- **E2EL (End-to-End Latency)**: Total time from request to complete response

## Tips for Optimal Benchmarking

1. **NUMA Awareness**: When using multiple NUMA nodes, be aware of memory access patterns
2. **Core Allocation**: For hyper-threaded CPUs, use only one thread per core for vLLM
3. **KV Cache Sizing**: Adjust the KV cache size based on your available RAM
4. **Benchmark Parameters**: Test with various concurrency levels and input/output lengths
5. **Dedicated Cores**: Run the benchmark script on separate cores from the vLLM processes

## Troubleshooting

- If the initial test run fails, verify that all arguments are correctly specified
- For memory errors, reduce the KV cache size or number of concurrent requests
- If using a corporate network, ensure proxy settings are correctly configured
- Check Docker logs for any container-specific issues

## References

- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [LLMPerf Repository](https://github.com/ray-project/llmperf)
