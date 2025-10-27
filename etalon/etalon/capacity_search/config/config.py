import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Optional


def _get_hash(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


@dataclass
class ServerConfig:
    openai_server_engine: Optional[str] = None
    openai_api_url: Optional[str] = None
    openai_api_key: Optional[str] = None

    def get_key(self):
        return (
            f"{self.openai_server_engine}_{self.openai_api_url}_{self.openai_api_key}"
        )

    def get_human_readable_name(self):
        return f"Server engine: {self.openai_server_engine}, URL: {self.openai_api_url}"

    def to_config_dict(self):
        return {
            "openai_server_engine": self.openai_server_engine,
            "openai_api_url": self.openai_api_url,
            "openai_api_key": self.openai_api_key,
        }


@dataclass
class ModelConfig:
    name: str
    identifier: str
    tokenizer: Optional[str] = None

    def get_key(self):
        return f"{self.name}"

    def get_human_readable_name(self):
        return f"Model: {self.name}, Tokenizer: {self.tokenizer}"

    def to_config_dict(self):
        return {"model_name": self.identifier, "tokenizer_name": self.tokenizer}

    def to_args(self):
        command = f"--client_config_model {self.identifier}"
        if self.tokenizer:
            command += f" --client_config_tokenizer {self.tokenizer}"
        return command


@dataclass
class RequestGeneratorConfig:
    start_qps: float
    request_interval_generator_provider: str
    request_length_generator_provider: str
    request_generator_max_tokens: Optional[int] = None
    gamma_request_interval_generator_cv: Optional[float] = None
    trace_request_interval_generator_trace_file: Optional[str] = None
    trace_request_interval_generator_start_time: Optional[str] = None
    trace_request_interval_generator_end_time: Optional[str] = None
    trace_request_interval_generator_time_scale_factor: Optional[float] = None
    trace_request_length_generator_trace_file: Optional[str] = None
    trace_request_length_generator_prefill_scale_factor: Optional[float] = None
    trace_request_length_generator_decode_scale_factor: Optional[float] = None
    fixed_request_generator_prefill_tokens: Optional[int] = None
    fixed_request_generator_decode_tokens: Optional[int] = None
    synthetic_request_generator_min_tokens: Optional[int] = None
    synthetic_request_generator_prefill_to_decode_ratio: Optional[float] = None
    zipf_request_length_generator_theta: Optional[float] = None
    zipf_request_length_generator_scramble: Optional[bool] = None
    seed: Optional[int] = 42
    trace_file_name: Optional[str] = None

    def get_key(self):
        key = f"{self.request_interval_generator_provider}_{self.request_length_generator_provider}_{self.start_qps}"
        if self.request_interval_generator_provider == "gamma":
            key += f"_{self.gamma_request_interval_generator_cv}"
        return key

    def get_human_readable_name(self):
        return f"Start QPS: {self.start_qps}, Request interval generator: {self.request_interval_generator_provider}, Request length generator: {self.request_length_generator_provider}"

    def to_config_dict(self):
        config_dict = {
            "request_interval_generator_config_type": self.request_interval_generator_provider,
            "request_length_generator_config_type": self.request_length_generator_provider,
            "seed": self.seed,
        }
        if self.request_interval_generator_provider == "gamma":
            config_dict["gamma_request_interval_generator_config_cv"] = (
                self.gamma_request_interval_generator_cv
            )
        elif self.request_interval_generator_provider == "trace":
            config_dict["trace_request_interval_generator_config_trace_file"] = (
                self.trace_request_interval_generator_trace_file
            )
            config_dict["trace_request_interval_generator_config_start_time"] = (
                self.trace_request_interval_generator_start_time
            )
            config_dict["trace_request_interval_generator_config_end_time"] = (
                self.trace_request_interval_generator_end_time
            )
            config_dict["trace_request_interval_generator_config_time_scale_factor"] = (
                self.trace_request_interval_generator_time_scale_factor
            )

        if self.request_length_generator_provider == "trace":
            config_dict["trace_request_length_generator_config_trace_file"] = (
                self.trace_request_length_generator_trace_file
            )
            config_dict[
                "trace_request_length_generator_config_prefill_scale_factor"
            ] = self.trace_request_length_generator_prefill_scale_factor
            config_dict["trace_request_length_generator_config_decode_scale_factor"] = (
                self.trace_request_length_generator_decode_scale_factor
            )
            config_dict["trace_request_length_generator_config_max_tokens"] = (
                self.request_generator_max_tokens
            )
        elif self.request_length_generator_provider == "fixed":
            config_dict["fixed_request_length_generator_config_prefill_tokens"] = (
                self.fixed_request_generator_prefill_tokens
            )
            config_dict["fixed_request_length_generator_config_decode_tokens"] = (
                self.fixed_request_generator_decode_tokens
            )
            config_dict["fixed_request_length_generator_config_max_tokens"] = (
                self.request_generator_max_tokens
            )
        elif self.request_length_generator_provider == "synthetic":
            config_dict["uniform_request_length_generator_config_min_tokens"] = (
                self.synthetic_request_generator_min_tokens
            )
            config_dict["uniform_request_length_generator_config_max_tokens"] = (
                self.request_generator_max_tokens
            )
            config_dict[
                "uniform_request_length_generator_config_prefill_to_decode_ratio"
            ] = self.synthetic_request_generator_prefill_to_decode_ratio
        elif self.request_length_generator_provider == "zipf":
            config_dict["zipf_request_length_generator_config_theta"] = (
                self.zipf_request_length_generator_theta
            )
            config_dict["zipf_request_length_generator_config_scramble"] = (
                self.zipf_request_length_generator_scramble
            )
            config_dict["zipf_request_length_generator_config_max_tokens"] = (
                self.request_generator_max_tokens
            )
        return config_dict

    def to_args(self):
        config_dict = self.to_config_dict()
        args = []
        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    args.append(f"--{key}")
                else:
                    args.append(f"--{key} {value}")
        return " ".join(args)


@dataclass
class ClientConfig:
    num_clients: Optional[int] = None
    num_concurrent_requests_per_client: Optional[int] = None
    timeout: Optional[int] = None
    max_num_completed_requests: Optional[int] = None
    additional_sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None

    def to_config_dict(self):
        return {
            "client_config_num_clients": self.num_clients,
            "client_config_num_concurrent_requests_per_client": self.num_concurrent_requests_per_client,
            "timeout": self.timeout,
            "max_completed_requests": self.max_num_completed_requests,
            "client_config_additional_sampling_params": self.additional_sampling_params,
            "client_config_llm_api": self.llm_api,
        }

    def to_args(self):
        config_dict = self.to_config_dict()
        args = []
        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    args.append(f"--{key}")
                else:
                    args.append(f"--{key} {value}")
        return " ".join(args)

    def get_key(self):
        return f"{self.num_clients}_{self.timeout}_{self.max_num_completed_requests}_{self.llm_api}"

    def to_human_readable_name(self):
        return f"Num ray clients: {self.num_clients}, Num concurrent requests per client: {self.num_concurrent_requests_per_client}, Timeout: {self.timeout}, Max num completed requests: {self.max_num_completed_requests}, LLM API: {self.llm_api}"


@dataclass
class JobConfig:
    def __init__(
        self,
        model_config: ModelConfig,
        request_generator_config: RequestGeneratorConfig,
        client_config: ClientConfig,
        server_config: ServerConfig,
    ):
        self.model_config = model_config
        self.request_generator_config = request_generator_config
        self.client_config = client_config
        self.server_config = server_config

        self.start_qps = self.request_generator_config.start_qps

    def get_key(self):
        config_keys = [
            self.model_config.get_key(),
            self.request_generator_config.get_key(),
            self.client_config.get_key(),
            self.server_config.get_key(),
        ]

        return "_".join(config_keys)

    def get_human_readable_name(self):
        substrings = [
            self.model_config.get_human_readable_name(),
            self.request_generator_config.get_human_readable_name(),
            self.client_config.to_human_readable_name(),
            self.server_config.get_human_readable_name(),
            f"Hash: {_get_hash(self.get_key())}",
        ]
        return ", ".join(substrings)

    def to_config_dict(self):
        return {
            **self.model_config.to_config_dict(),
            **self.request_generator_config.to_config_dict(),
            **self.client_config.to_config_dict(),
            **self.server_config.to_config_dict(),
        }

    def to_args(self):
        model_args = self.model_config.to_args()
        request_generator_args = self.request_generator_config.to_args()
        request_args = self.client_config.to_args()
        return f"{model_args} {request_generator_args} {request_args}"

    @classmethod
    def generate_job_configs(cls, config: dict):
        job_configs = []
        for (
            model_config,
            request_generator_config,
            client_config,
            server_config,
        ) in product(
            config["models"],
            config["request_generator_configs"],
            config["client_configs"],
            config["servers"],
        ):
            model_config = ModelConfig(**model_config)
            request_generator_config = RequestGeneratorConfig(
                **request_generator_config
            )
            client_config = ClientConfig(**client_config)
            server_config = ServerConfig(**server_config)

            job_config = cls(
                model_config,
                request_generator_config,
                client_config,
                server_config,
            )
            job_configs.append(job_config)

        return job_configs

    def __str__(self):
        return self.get_human_readable_name()


@dataclass
class BenchmarkConfig:
    output_dir: str
    qps: float
    should_use_given_dir: bool = True
    ttft_deadline: Optional[float] = None
    tbt_deadline: Optional[float] = None
    ttft_slack: Optional[float] = None
    wandb_group: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    should_write_metrics: Optional[bool] = True
    use_predictions_for_ttft: Optional[bool] = False
    predictor_dir: Optional[str] = None

    def to_config_dict(self):
        return {
            "metrics_config_output_dir": self.output_dir,
            "gamma_request_interval_generator_config_qps": self.qps,
            "poisson_request_interval_generator_config_qps": self.qps,
            "metrics_config_should_use_given_dir": self.should_use_given_dir,
            "deadline_config_ttft_deadline": self.ttft_deadline,
            "deadline_config_tbt_deadline": self.tbt_deadline,
            "deadline_config_ttft_slack": self.ttft_slack,
            "metrics_config_wandb_group": self.wandb_group,
            "metrics_config_wandb_project": self.wandb_project,
            "metrics_config_wandb_run_name": self.wandb_run_name,
            "metrics_config_should_write_metrics": self.should_write_metrics,
            "prefill_profiler_config_use_predictions_for_ttft": self.use_predictions_for_ttft,
            "prefill_profiler_config_predictor_dir": self.predictor_dir,
        }

    def to_args(self):
        config_dict = self.to_config_dict()
        args = []
        for key, value in config_dict.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        args.append(f"--{key}")
                    else:
                        args.append(f"--no-{key}")
                else:
                    args.append(f"--{key} {value}")
        return " ".join(args)

    def get_run_id(self):
        return f"{self.qps}"

    def get_run_dir(self):
        return self.output_dir

    def to_human_readable_name(self):
        return f"QPS: {self.qps}, Run id: {self.get_run_id()}"
