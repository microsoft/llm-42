"""
This file contains the wrapper for the benchmarking.
"""

import os
import subprocess

from etalon.capacity_search.config.config import BenchmarkConfig, JobConfig
from etalon.logger import init_logger

logger = init_logger(__name__)


def setup_api_environment(
    openai_api_key=None,
    openai_api_url=None,
):
    """Set up environment variables for OpenAI API"""
    assert openai_api_key is not None, "OpenAI API key is required"
    assert openai_api_url is not None, "OpenAI port is required"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = openai_api_url


def run(
    job_config: JobConfig,
    benchmark_config: BenchmarkConfig,
):
    """Main function to run benchmark"""

    setup_api_environment(
        openai_api_key=job_config.server_config.openai_api_key,
        openai_api_url=job_config.server_config.openai_api_url,
    )

    benchmark_command = f"python -m etalon.run_benchmark {job_config.to_args()} {benchmark_config.to_args()}"
    logger.info(f"Running benchmark with command: {benchmark_command}")
    benchmark_process = subprocess.Popen(benchmark_command, shell=True)
    benchmark_process.wait()
    logger.info("Benchmark finished")
