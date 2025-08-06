import json
from pathlib import Path
from typing import List
import torch
from transformers import LogitsProcessorList, LogitsProcessor
from selective_sampling.logit_processors.base import BaseLogitsProcessor
from selective_sampling.reward_modeling.classifier import (
    TemperatureClassifierConfig,
    TemperatureClassifier,
    BaseTemperatureClassifier,
    DigitsTemperatureClassifier,
    EntropyTemperatureClassifier,
    NgramEmbeddingsClassifierConfig,
)

import logging

from selective_sampling.utils import process_temperature

logging.basicConfig(level=logging.INFO)


def load_greedy_samples(path):
    logging.info(f"Loading greedy samples from {path}")
    samples = list(Path(path).glob("**/samples_*.jsonl"))
    if len(samples) != 1:
        logging.error(f"No samples found in {path}")
        exit(0)
    sample_file = samples[0]

    samples = []
    with open(sample_file, "r") as f:
        for sample in f:
            samples.append(json.loads(sample))

    return samples


class TokenImportanceLogitProcessor:
    def __init__(self, token_importance_config, greedy_samples=None, **kwargs):
        self.token_importance_config = token_importance_config

        self.log = {}  # input_ids -> top-k token logits
        self.log_top_k = 10

        if self.token_importance_config.step == -1:
            self.greedy_samples = None
        else:
            if greedy_samples is None:
                self.greedy_samples = load_greedy_samples(
                    self.token_importance_config.output_path_greedy
                )
            else:
                self.greedy_samples = greedy_samples

            self.prompt_ids_to_greedy_samples = {}

            for sample in self.greedy_samples:
                prompt_ids = tuple(sample["extra"][0]["prompt_token_ids"])
                token_ids = tuple(sample["extra"][0]["token_ids"])
                top_tokens: List[List[int]] = sample["extra"][0]["top_tokens"]
                self.prompt_ids_to_greedy_samples[prompt_ids] = torch.tensor(
                    token_ids
                ).to("cuda:0"), torch.tensor(top_tokens).to("cuda:0")

    def __call__(self, prompt_tokens_ids, input_ids, logits):
        """
        Greedy decoding with a switch to top-2 token at a certain step
        Used to estimate the importance of the top-1 tokens
        """
        # TODO: can we use batched logit processor? seems that vllm does not support it
        # Maybe we will switch for guided decoding then?

        current_token_id = len(
            input_ids
        )  # for vllm input ids are already for output_token_ids, without the prompt

        # if position == -1, we will log all logits (before softmax)
        if self.token_importance_config.step == -1:
            _top_k = logits.topk(k=self.log_top_k, dim=-1)
            top_k_logits = _top_k.values
            top_k_tokens = _top_k.indices
            self.log[prompt_tokens_ids + input_ids] = {
                "top_k_logits": top_k_logits,
                "top_k_token_ids": top_k_tokens,
            }
            # logging.info(f"Logging logits for prompt {prompt_tokens_ids + input_ids}")

        else:
            # synchronize with greedy decoding
            # (token_ids, top_tokens)
            greedy_token_ids, top_tokens = self.prompt_ids_to_greedy_samples[
                prompt_tokens_ids
            ]

            if current_token_id < self.token_importance_config.step:
                # copy greedy token ids
                selected_token_id = greedy_token_ids[current_token_id]
                # scatter -inf everwhere
                logits.fill_(float("-inf"))
                logits.scatter_(dim=-1, index=selected_token_id, value=0.0)
            elif current_token_id == self.token_importance_config.step:
                # we need to set the logit of top-1 token to -inf
                # 1) Find the top-k indices along the last dimension

                assert self.token_importance_config.token_rank < len(
                    top_tokens[current_token_id]
                )
                selected_token_id = top_tokens[current_token_id][
                    self.token_importance_config.token_rank
                ]
                # scatter -inf everwhere
                logits.fill_(float("-inf"))
                logits.scatter_(dim=-1, index=selected_token_id, value=0.0)

                # k is given by config.token_rank
                # top_k_indices = logits.topk(
                #     k=self.token_importance_config.token_rank, dim=-1
                # ).indices

                # # 2) Scatter -inf at those indices
                # logits.scatter_(dim=-1, index=top_k_indices, value=float("-inf"))
            else:
                return logits

        return logits


class TokenImportanceConfig:
    def __init__(self, step: int, token_rank: int = 2, output_path_greedy=None):
        self.step = step  # current step of the decoding where we switch to top-2 token
        self.token_rank = (
            token_rank  # top-i token to consider in top-k token importance tree search
        )
        self.output_path_greedy = output_path_greedy  # path to save greedy decoding


def parse_token_importance_classifier_sampling(
    kwargs: dict,
) -> TokenImportanceConfig | None:
    step = kwargs.pop("token_importance", None)
    if step is None:
        return None
    token_rank = kwargs.pop("token_rank", None)
    output_path_greedy = kwargs.pop("output_path_greedy", None)
    return TokenImportanceConfig(
        step=step, token_rank=token_rank, output_path_greedy=output_path_greedy
    )


class DummyTemperatureClassifier:
    def __init__(self, *args, **kwargs):
        self._max_temperature = 1.0
        self.config = None

    def set_max_temperature(self, temperature):
        self._max_temperature = temperature

    def __call__(self, input_ids, logits):
        return self._max_temperature


class DoubleTemperature:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self._max_temperature = 1.0
        self._support_temperature = 1.0
        if "override__support_temperature" in kwargs:
            self._support_temperature = kwargs.pop("override__support_temperature")

    def set_max_temperature(self, temperature):
        self._max_temperature = temperature

    def set_support_temperature(self, temperature):
        self._support_temperature = temperature

    def __call__(self, input_ids, logits):
        return self._max_temperature


def get_temperature_classifier(configs, experiment_name, kwargs):
    if experiment_name == "embeddings":
        # config = TemperatureClassifierConfig(configs=configs)
        # temperature_classifier = TemperatureClassifier(config, device="cuda")
        # print(temperature_classifier, flush=True)
        # return temperature_classifier
        raise NotImplementedError
    elif experiment_name == "embeddings_ours":  # ngram like classifier
        classifier_path = kwargs.pop("classifier_path", None)
        config = TemperatureClassifierConfig(configs=configs)
        temperature_classifier = NgramEmbeddingsClassifierConfig(
            config, classifier_path, device="cuda"
        )
        print(temperature_classifier, flush=True)
        return temperature_classifier
    elif experiment_name == "digits":
        config = TemperatureClassifierConfig(configs=configs)
        return DigitsTemperatureClassifier(config=config, device="cpu")
    elif experiment_name == "entropy":
        theta = kwargs.pop("theta", None)
        N = 0.8
        return EntropyTemperatureClassifier(
            config=None, theta=theta, N=N, device="cuda"
        )
    elif experiment_name == "dummy":
        return DummyTemperatureClassifier()
    else:
        raise ValueError(f"Unknown experiment name {experiment_name}")


def parse_temperature_classifier_sampling(kwargs: dict) -> BaseTemperatureClassifier:
    configs = kwargs.pop("temperature_classifier_configs", None)
    experiment_name = kwargs.pop("temperature_classifier_experiment", None)

    if configs is None:
        return None

    assert (
        experiment_name is not None
    ), "Experiment name should be provided for dynamic temperature classifier: see get_temperature_classifier"

    logging.info(
        f"Loading temperature classifier with configs {configs} and experiment name {experiment_name}"
    )

    classifier = get_temperature_classifier(configs, experiment_name, kwargs)
    logging.info(f"Loaded temperature classifier {classifier}")
    return classifier


class TemperatureClassifierLogitProcessor:
    def __init__(
        self, temperature_classifier: BaseTemperatureClassifier, temperature: float
    ):
        self.temperature_classifier = temperature_classifier
        self.temperature_classifier.set_max_temperature(
            temperature
        )  # setting maximal temperature for the classifier
        self.eps = 1e-5

    def __call__(self, input_ids, logits):

        temperature = self.temperature_classifier(input_ids, logits)

        return process_temperature(
            temperature=temperature,
            logits=logits,
            eps=self.eps,
        )
