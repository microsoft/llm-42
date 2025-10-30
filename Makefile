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


MODEL ?= meta-llama/Meta-Llama-3-8B 
BACKEND ?= flashinfer
TOKENS = 8192
BATCH ?= 2048
PROFILE = #--profile
OVERRIDE = #--json-model-override-args '{"num_hidden_layers": 1}'
OUTPUT ?= output/offline2_Q3-4B_
INPUT_TOKEN ?= 1024
OUTPUT_TOKEN ?= 1024
EXTRA ?= #--enable-deterministic-inference
POST ?= 
PRE_ENVS ?= 

run_offline:
	python3 -m sglang.bench_offline_throughput --model-path ${MODEL} ${PROFILE} ${OVERRIDE} \
		--num-prompts ${BATCH} --attention-backend ${BACKEND} \
		--dataset-name random --random-input ${INPUT_TOKEN} --random-output ${OUTPUT_TOKEN} ${EXTRA} > ${OUTPUT}${BACKEND}_b${BATCH}_in${INPUT_TOKEN}_out${OUTPUT_TOKEN}_${POST}.txt 2>&1;

run_offline_vllm:
	${PRE_ENVS} vllm bench throughput --dataset-name=random --input-len=1024 --output-len=256 --num-prompts=${BATCH} \
		--model=${MODEL} > ${OUTPUT}b${BATCH}_in${INPUT_TOKEN}_out${OUTPUT_TOKEN}_${POST}.txt 2>&1

# Bitmask values for deterministic inference modes:
# 1   = On with defaults (det matmul, det rmsnorm, det attention)
# 2   = Use kernel matmul
# 4   = Use split-stream matmul
# 32  = Use non-det matmul
# 64  = Use non-det rmsnorm
# 128 = Use non-det attention
#
# Common combinations:
# 1   = Full deterministic (default)
# 2   = Deterministic with kernel matmul
# 3   = Deterministic with kernel matmul (1+2)
# 4   = Deterministic with split-stream matmul
# 5   = Deterministic with split-stream matmul (1+4)
# 6   = Kernel + split-stream matmul (2+4)
# 32  = Non-deterministic matmul only
# 64  = Non-deterministic rmsnorm only
# 96  = Non-deterministic matmul + rmsnorm (32+64)
# 128 = Non-deterministic attention only
# 160 = Non-deterministic matmul + attention (32+128)
# 192 = Non-deterministic rmsnorm + attention (64+128)
# 194 = Non-deterministic matmul + rmsnorm + attention, kernel matmul (2+32+64+128)
# 224 = All non-deterministic (32+64+128)

MODE ?= 1
run_offline_det:
	$(MAKE) run_offline EXTRA="--enable-deterministic-inference=${MODE}" POST="det${MODE}_";

# All deterministic modes for comprehensive testing
MODES_DET = 1 2 3 4 5 6
# All non-deterministic component modes
MODES_NONDET = 32 64 96 128 160 192 224
# Hybrid modes (some components deterministic, some not)
MODES_HYBRID = 194
# All modes combined
MODES_ALL = $(MODES_DET) $(MODES_NONDET) $(MODES_HYBRID)

run_test_perf:
	mkdir -p output
	export SGLANG_TORCH_PROFILER_DIR=$$(pwd)/profile_output; \
	for token in ${TOKENS}; do \
		$(MAKE) run_offline INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}; \
		$(MAKE) run_offline_det INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token} MODE=96; \
	done
	$(MAKE) extract_test_deterministic_perf;

run_test_all_modes: ## Run benchmarks with all deterministic mode combinations
	mkdir -p output
	export SGLANG_TORCH_PROFILER_DIR=$$(pwd)/profile_output; \
	for token in ${TOKENS}; do \
		echo "Running baseline (non-deterministic) with INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}"; \
		$(MAKE) run_offline INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}; \
		for mode in $(MODES_ALL); do \
			echo "Running deterministic mode=$${mode} with INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}"; \
			$(MAKE) run_offline_det INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token} MODE=$${mode}; \
		done \
	done

run_test_det_modes: ## Run benchmarks with fully deterministic modes only
	mkdir -p output
	export SGLANG_TORCH_PROFILER_DIR=$$(pwd)/profile_output; \
	for token in ${TOKENS}; do \
		echo "Running baseline (non-deterministic) with INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}"; \
		$(MAKE) run_offline INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}; \
		for mode in $(MODES_DET); do \
			echo "Running deterministic mode=$${mode} with INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}"; \
			$(MAKE) run_offline_det INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token} MODE=$${mode}; \
		done \
	done

run_test_nondet_modes: ## Run benchmarks with non-deterministic component modes
	mkdir -p output
	export SGLANG_TORCH_PROFILER_DIR=$$(pwd)/profile_output; \
	for token in ${TOKENS}; do \
		echo "Running baseline (non-deterministic) with INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}"; \
		$(MAKE) run_offline INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}; \
		for mode in $(MODES_NONDET); do \
			echo "Running mode=$${mode} (non-det components) with INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token}"; \
			$(MAKE) run_offline_det INPUT_TOKEN=$${token} OUTPUT_TOKEN=$${token} MODE=$${mode}; \
		done \
	done

extract_test_deterministic_perf:
	@echo "Extracting and displaying benchmark results:"
	for file in ${OUTPUT}${BACKEND}_b${BATCH}*.txt; do \
		echo "Results from $$file:"; \
		grep -A11 "Offline Throughput Benchmark Result" $$file; \
	done

#	python3 -m sglang.bench_offline_throughput --model-path ${MODEL} --num-prompts 256 --attention-backend flashinfer --dataset-name random --random-input 8192 --random-output 8192 | grep -A11 "Offline Throughput Benchmark Result"
#	python3 -m sglang.bench_offline_throughput --model-path ${MODEL} --num-prompts 256 --attention-backend flashinfer --dataset-name random --random-input 8192 --random-output 8192 --enable-deterministic-inference | grep -A11 "Offline Throughput Benchmark Result"

# Plotted experiments
INTRO_OUT ?= output/intro/
FIGURES = figures/
figure_1_sgl:
	mkdir -p ${INTRO_OUT};
	$(MAKE) run_offline INPUT_TOKEN=1024 OUTPUT_TOKEN=256 OUTPUT=${INTRO_OUT} POST=nondet;
	$(MAKE) run_offline_det INPUT_TOKEN=1024 OUTPUT_TOKEN=256 OUTPUT=${INTRO_OUT} MODE=1 POST=detbase;
	$(MAKE) run_offline_det INPUT_TOKEN=1024 OUTPUT_TOKEN=256 OUTPUT=${INTRO_OUT} MODE=66 POST=ours;

	$(MAKE) run_offline INPUT_TOKEN=1024 OUTPUT_TOKEN=512 OUTPUT=${INTRO_OUT} POST=nondet;
	$(MAKE) run_offline_det INPUT_TOKEN=1024 OUTPUT_TOKEN=512 OUTPUT=${INTRO_OUT} MODE=1 POST=detbase;
	$(MAKE) run_offline_det INPUT_TOKEN=1024 OUTPUT_TOKEN=512 OUTPUT=${INTRO_OUT} MODE=66 POST=ours;

	$(MAKE) run_offline INPUT_TOKEN=2048 OUTPUT_TOKEN=1024 OUTPUT=${INTRO_OUT} POST=nondet;
	$(MAKE) run_offline_det INPUT_TOKEN=2048 OUTPUT_TOKEN=1024 OUTPUT=${INTRO_OUT} MODE=1 POST=detbase;
	$(MAKE) run_offline_det INPUT_TOKEN=2048 OUTPUT_TOKEN=1024 OUTPUT=${INTRO_OUT} MODE=66 POST=ours;

plot_figure_1:
	mkdir -p ${FIGURES}
	python3 batch_invariant/scripts/plot_introfig.py ${INTRO_OUT} -o ${FIGURES}/figure_1.png

figure_1_vllm:
	mkdir -p ${INTRO_OUT};
	$(MAKE) run_offline_vllm INPUT_TOKEN=1024 OUTPUT_TOKEN=256 OUTPUT=${INTRO_OUT} POST=vllm_nondet;
	$(MAKE) run_offline_vllm INPUT_TOKEN=1024 OUTPUT_TOKEN=256 OUTPUT=${INTRO_OUT} POST=vllm_det PRE_ENVS="VLLM_BATCH_INVARIANT=1";
	VLLM_BATCH_INVARIANT=0 VLLM_ATTENTION_BACKEND=FLEX_ATTENTION vllm bench throughput --dataset-name=random --input-len=1024 --output-len=256 --num-prompts=${BATCH} \
		--model=${MODEL} --enforce_eager > ${INTRO_OUT}b${BATCH}_in${INPUT_TOKEN}_out${OUTPUT_TOKEN}_vllmnondet.txt 2>&1
	VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLEX_ATTENTION vllm bench throughput --dataset-name=random --input-len=1024 --output-len=256 --num-prompts=${BATCH} \
		--model=${MODEL} --enforce_eager > ${INTRO_OUT}b${BATCH}_in${INPUT_TOKEN}_out${OUTPUT_TOKEN}_vllmdet.txt 2>&1

figure_prefill:
	mkdir -p output/figure_3/
	python batch_invariant/bench_flashinfer.py > output/figure_3/figure_3_results.txt 2>&1

plot_figure_prefill:
	python batch_invariant/scripts/plot_flashinfer_prefill.py output/figure_3/figure_3_results.txt

figure_matmul:
	mkdir -p output/figure_4/
	python batch_invariant/bench_llama3_ops.py > output/figure_4/figure_4_results.txt 2>&1

plot_matmul:
	python batch_invariant/scripts/plot_llama_matmul.py output/figure_4/figure_4_results.txt

figure_rmsnorm:
	mkdir -p output/figure_rmsnorm/
	python batch_invariant/bench_rmsnorm.py > output/figure_rmsnorm/figure_rmsnorm_results.txt 2>&1

plot_rmsnorm:
	python batch_invariant/scripts/plot_rmsnorm.py output/figure_rmsnorm/figure_rmsnorm_results.txt

figure_eval:
	cd verify_bench_per_request; \
	sh test_all_traces_llama.sh;

figure_eval_offline:
	mkdir -p output/eval_offline/;
	$(MAKE) figure_1_sgl MODEL=Qwen/Qwen3-8B-Base INTRO_OUT=output/eval_offline/

plot_eval_offline:
	mkdir -p ${FIGURES}
	python3 batch_invariant/scripts/plot_introfig.py output/eval_offline/ -o ${FIGURES}/figure_eval_offline.png --title "Qwen3-8B-Base, H200"
