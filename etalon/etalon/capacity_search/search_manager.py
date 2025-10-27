import argparse
from functools import partial
from multiprocessing import Pool

from etalon.capacity_search.capacity_search import CapacitySearch
from etalon.capacity_search.config.config import JobConfig
from etalon.logger import init_logger

logger = init_logger(__name__)


def run_search(
    job_config: JobConfig,
    args: argparse.Namespace,
):
    capacity_search = CapacitySearch(
        job_config,
        args,
    )
    return capacity_search.search()


class SearchManager:
    def __init__(
        self,
        args: argparse.Namespace,
        config: dict,
    ):
        self.args = args
        self.config = config

    def run(self):
        job_configs = JobConfig.generate_job_configs(self.config)
        num_jobs = len(job_configs)
        logger.info(f"Running {num_jobs} jobs")

        with Pool(processes=num_jobs) as capacity_search_pool:
            run_search_partial = partial(run_search, args=self.args)  # Pre-fill `args`
            all_results = capacity_search_pool.map(run_search_partial, job_configs)

        return all_results
