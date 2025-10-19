.PHONY: check-deps install-deps format update help

# Show help for each target
help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check-deps: ## Check and install required Python formatting dependencies
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && pip install isort)
	@command -v black >/dev/null 2>&1 || (echo "Installing black..." && pip install black)

install-deps: ## Install Python formatting tools (isort and black)
	pip install isort black

format: check-deps ## Format modified Python files using isort and black
	@echo "Formatting modified Python files..."
	git diff --name-only --diff-filter=M | grep '\.py$$' | xargs -I {} sh -c 'isort {} && black {}'

FILES_TO_UPDATE = docker/Dockerfile.rocm \
                 python/pyproject.toml \
                 python/pyproject_other.toml \
                 python/sglang/version.py \
                 docs/developer_guide/setup_github_runner.md \
                 docs/get_started/install.md \
                 docs/platforms/amd_gpu.md \
                 docs/platforms/ascend_npu.md \
				 benchmark/deepseek_v3/README.md

update: ## Update version numbers across project files. Usage: make update <new_version>
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Version required. Usage: make update <new_version>"; \
		exit 1; \
	fi
	@OLD_VERSION=$$(grep "version" python/sglang/version.py | cut -d '"' -f2); \
	NEW_VERSION=$(filter-out $@,$(MAKECMDGOALS)); \
	echo "Updating version from $$OLD_VERSION to $$NEW_VERSION"; \
	for file in $(FILES_TO_UPDATE); do \
		if [ "$(shell uname)" = "Darwin" ]; then \
			sed -i '' -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \
		else \
			sed -i -e "s/$$OLD_VERSION/$$NEW_VERSION/g" $$file; \
		fi \
	done; \
	echo "Version update complete"

%:
	@:

conda:
	conda install -c nvidia cuda=12.8
	export CUDA_HOME=/m-coriander/coriander/akkamath/miniconda3/envs/sglang2/targets/x86_64-linux/
	export CMAKE_PREFIX_PATH=$CUDA_HOME:$CMAKE_PREFIX_PATH
	export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH


MODEL = Qwen/Qwen3-4B
BACKEND ?= flashinfer
TOKENS = 1024
BATCH ?= 256
PROFILE = #--profile
OVERRIDE = #--json-model-override-args '{"num_hidden_layers": 1}'
OUTPUT ?= output/offline2_Q3-4B_
TOKEN ?= 1024
EXTRA ?= #--enable-deterministic-inference
POST ?= 

run_offline:
	python3 -m sglang.bench_offline_throughput --model-path ${MODEL} ${PROFILE} ${OVERRIDE} \
		--num-prompts ${BATCH} --attention-backend ${BACKEND} \
		--dataset-name random --random-input ${TOKEN} --random-output ${TOKEN} ${EXTRA} > ${OUTPUT}${POST}${BACKEND}_b${BATCH}_${TOKEN}.txt;

run_offline_det:
	$(MAKE) run_offline EXTRA="--enable-deterministic-inference" POST="det_";

run_test_perf:
	mkdir -p output
	export SGLANG_TORCH_PROFILER_DIR=$$(pwd)/profile_output; \
	for token in ${TOKENS}; do \
		$(MAKE) run_offline TOKEN=$${token}; \
		$(MAKE) run_offline_det TOKEN=$${token}; \
	done

extract_test_deterministic_perf:
	@echo "Extracting and displaying benchmark results:"
	for file in ${OUTPUT}*.txt; do \
		echo "Results from $$file:"; \
		grep -A11 "Offline Throughput Benchmark Result" $$file; \
	done

#	python3 -m sglang.bench_offline_throughput --model-path ${MODEL} --num-prompts 256 --attention-backend flashinfer --dataset-name random --random-input 8192 --random-output 8192 | grep -A11 "Offline Throughput Benchmark Result"
#	python3 -m sglang.bench_offline_throughput --model-path ${MODEL} --num-prompts 256 --attention-backend flashinfer --dataset-name random --random-input 8192 --random-output 8192 --enable-deterministic-inference | grep -A11 "Offline Throughput Benchmark Result"
