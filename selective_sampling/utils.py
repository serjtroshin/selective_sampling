from typing import Tuple
import warnings
import yaml
import logging
from pathlib import Path

from selective_sampling.configs import LogitProcessorConfig
from transformers import LogitsProcessorList

warnings.simplefilter("once")
CONFIGS = "selective_sampling/configs"

logger = logging.getLogger(__name__)


def get_tasks_path():
    return Path(__file__).parent / "tasks"


def load_config(config_name: str) -> dict:
    current_dir = Path(__file__).parent.parent
    config_name = current_dir / config_name
    with open(str(config_name), "r") as f:
        config = yaml.safe_load(f)
    assert config is not None, f"Config file {str(config_name)} is empty"
    return config


def parse_logit_processor_kwargs(kwargs: dict) -> Tuple[dict, LogitsProcessorList]:
    logit_processor = kwargs.pop("logit_processor", None)
    if logit_processor is None:
        return kwargs, LogitsProcessorList()

    logit_processor_list = []

    # each logit processor implements its own required kwargs via from_kwargs method
    # don't forget to remove these kwargs from the kwargs dict
    if logit_processor == "dummy":
        from selective_sampling.logit_processors import DummyLogitsProcessorList

        logit_processor_list = DummyLogitsProcessorList.from_kwargs(kwargs)
    else:
        raise ValueError(f"Unknown logit processor type {logit_processor}")
    return kwargs, logit_processor_list


def parse_sampling_config(kwargs: dict) -> dict:
    sampling_mode = kwargs.pop("sampling_config", "greedy")  # config name
    sampling_kwargs = load_config(f"{CONFIGS}/sampling/{sampling_mode}.yaml")
    logger.info(str(kwargs))
    assert (
        "seed" in kwargs
    ), f"seed is required for sampling, 1 seed is used to sample 1 sample for each prompt, got {str(kwargs)}"
    sampling_kwargs["seed"] = kwargs["seed"]

    # override config
    for key, value in kwargs.items():
        if key.startswith("override__"):
            key = key.replace("override__", "")
            logger.info(f"Overriding sampling config: {key}={value}")
            sampling_kwargs[key] = value

    logger.info(f"Loaded sampling config: {sampling_kwargs}")

    return sampling_kwargs


def warn_about_new_args(generation_kwargs, extra_generation_kwargs):
    overrided_keys = generation_kwargs.keys() & extra_generation_kwargs.keys()
    new_keys = extra_generation_kwargs.keys() - generation_kwargs.keys()
    warnings.warn(
        f"Overrided generation kwargs: {[(key, extra_generation_kwargs[key]) for key in overrided_keys if key != 'tokenizer']}"
    )
    warnings.warn(
        f"New generation kwargs: {[(key, extra_generation_kwargs[key]) for key in new_keys if key != 'tokenizer']}"
    )


def process_temperature(temperature, logits, eps=1e-5):
    assert temperature >= 0.0, f"Temperature should be non-negative, got {temperature}"
    # print(f"Temperature: {temperature}", flush=True)
    if temperature < eps:
        argmax = logits.argmax(dim=-1)
        # set other logits to -inf
        logits.fill_(float("-inf"))
        logits.scatter_(dim=-1, index=argmax.unsqueeze(-1), value=0.0)
        # print("1. Temperature is too low, using argmax", flush=True)
        # print(f"Argmax: {argmax}", flush=True)
    else:
        logits = logits / temperature
        # print(f"2. using temperature: {temperature}", flush=True)
        # print("logits max/min", logits.max(), logits.min(), flush=True)
    return logits
