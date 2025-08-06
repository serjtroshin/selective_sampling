# add current dir to path
import argparse
from collections import defaultdict
import json
import os
import random
import sys
from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from safetensors.torch import load_model, save_model
from selective_sampling.reward_modeling.embedding_model import EmbeddingModel

import torch

sys.path.append(str(Path(__file__).parent))
CUR_PATH = Path(__file__).parent

from selective_sampling.reward_modeling.utils import read_yamls

# logging
import logging

logging.basicConfig(level=logging.INFO)


def parse_config(name: str):
    # name: str
    args = argparse.Namespace()

    # Config from YAML
    conf = {}
    configs = read_yamls(CUR_PATH / "configs")

    if "|" in name:
        for n in name.split("|"):
            conf.update(configs[n])
    else:
        conf.update(configs[name])

    print(conf)

    # convert to argparse.Namespace
    for key, value in conf.items():
        setattr(args, key, value)

    return args


class TemperatureClassifierConfig:
    def __init__(self, configs: str):
        self.config = parse_config(configs)


class BaseTemperatureClassifier:
    def __init__(self, config: TemperatureClassifierConfig, device: str = "cpu"):
        self.config = config.config if config is not None else None
        self.device = device
        self._max_temperature = 1.0

    def set_max_temperature(self, temperature: float):
        self._max_temperature = temperature

    def __call__(self, input_ids, logits: List[float]) -> float:
        raise NotImplementedError

    def get_top_1_token(self, logits) -> int:
        if isinstance(logits, list):
            logits = torch.tensor(logits)
        logits = logits.to(self.device)
        argmax_logits = logits.argmax(dim=-1)
        return argmax_logits


class NgramEmbeddingsClassifierConfig(BaseTemperatureClassifier):
    """
    Ngram like classifier: trained via embedding_model.py
    """

    def __init__(
        self, config: TemperatureClassifierConfig, classifier_path: str, device: str
    ):
        super().__init__(config, device)
        tokenizer = AutoTokenizer.from_pretrained(config.config.model_name)
        self.classifier = EmbeddingModel(
            torch.load(config.config.embeddings_path),
            tokenizer=tokenizer,
            hidden_dim=config.config.hidden_dim,
            n_layers=config.config.n_layers,
            k_past_tokens=config.config.k_past_tokens,
            do_conv_padding=False,
            pos_weight=1.0 if config.config.class_balance else None,
        )
        print(self.classifier)
        load_model(self.classifier, classifier_path)
        self.classifier.to(device)
        self.classifier.eval()

        # CHECK
        self.free_first_token = True

    @torch.no_grad()
    def __call__(self, input_ids, logits: List[float]) -> float:
        """applies classifier for last k tokens (with padding)
        we do padding here up to k_past_tokens
        """

        if self.free_first_token and len(input_ids) == 0:
            return self._max_temperature

        # 1. get last k tokens
        k_past_tokens = self.config.k_past_tokens
        if len(input_ids) < k_past_tokens:
            input_ids = [0] * (k_past_tokens - len(input_ids)) + list(input_ids)
        input_ids = (
            torch.tensor(input_ids[-k_past_tokens:]).to(self.device).unsqueeze(0)
        )  # unsqueeze for batch dim

        # 2. run through the classifier
        output = self.classifier(input_ids, input_ids, None)
        selective_sampling_logits = output.selective_sampling_logits.squeeze().cpu().item()

        if os.environ.get("DEBUG", False) == "True":

            def color_token(token, color):
                return f"\033[{color}m{token}\033[0m"

            top_1_token = self.get_top_1_token(logits)
            top_1_token = self.classifier.tokenizer.decode(top_1_token)
            if selective_sampling_logits >= 0:
                top_1_token = color_token(top_1_token, "31")

            print(
                self.classifier.tokenizer.decode(input_ids[0].detach().cpu().tolist()),
                " ",
                top_1_token,
                end="\n",
            )
            print("self._max_temperature", self._max_temperature)
            if selective_sampling_logits >= 0:
                print("using 0.0")
            else:
                print("using", self._max_temperature)
            input()
        # 1.0 means risky, 0.0 means safe
        if selective_sampling_logits >= 0:
            return 0.0
        else:
            return self._max_temperature


class TemperatureClassifier(BaseTemperatureClassifier):
    def __init__(self, config: TemperatureClassifierConfig, device: str):
        super().__init__(config, device)
        self.model = get_embedding_model(self.config, evaluate=True).to(device)
        self.tokenizer = self.model.tokenizer

    @torch.no_grad()
    def __call__(self, input_ids, logits: List[float]) -> float:
        """
        Returns the temperature for the current step
        Dynamic Temperature: T'(t) = T * (1 - x(t)), where x(t) is the output of the temperature classifier,
        x(t) = sigmoid(W * h(t) + b), probability of the greedy token being important
        """

        argmax_logits = self.get_top_1_token(logits)

        # get the prediction from the model
        _loss, prediction = self.model(argmax_logits)

        temperature = self._max_temperature * (1 - prediction.cpu().item())

        # print(self.tokenizer.decode(argmax_logits), prediction, temperature)
        return temperature


class DigitsTemperatureClassifier(BaseTemperatureClassifier):
    def __init__(self, config: TemperatureClassifierConfig, device: str):
        super().__init__(config, device)

        model_name = self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def model(self, input_ids: torch.Tensor):
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0)
        tokens = [self.tokenizer.decode(token) for token in input_ids]

        def is_digit(token):
            # test if all characters are digits
            digits = "0123456789"
            if all(char in digits for char in token):
                return True
            maths_chars = [
                "+",
                "-",
                "*",
                "/",
                "(",
                ")",
                "^",
                "%",
                "=",
                ">",
                "<",
                "!",
                "&",
                "|",
            ]
            if all(char in maths_chars for char in token):
                return True
            return False

        predictions = torch.tensor([int(is_digit(token)) for token in tokens]).squeeze(
            0
        )
        return None, predictions

    @torch.no_grad()
    def __call__(self, input_ids, logits: List[float]) -> float:

        argmax_logits = self.get_top_1_token(logits)
        predictions = self.model(argmax_logits)[1]
        token = self.tokenizer.decode(argmax_logits)
        # print(token, predictions)
        return self._max_temperature * (1.0 - predictions.cpu().item())


class EntropyTemperatureClassifier(BaseTemperatureClassifier):
    """
    https://arxiv.org/pdf/2403.14541.pdf
    Implementation of the Entropy-based
    Dynamic Temperature Sampling
    Entropy = - sum(p(x) * log(p(x)))
    T = T0 * N ^ {Theta / entropy}
    """

    def __init__(self, config: TemperatureClassifierConfig, device: str, theta, N=0.8):
        super().__init__(config, device)
        self.N = N  # default value from the paper
        assert (
            self.N <= 1.0
        )  # otherwise the temperature will increase (and go to inf for entropy=0)
        self.theta = theta

    @torch.no_grad()
    def __call__(self, input_ids, logits: List[float]) -> float:
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        if entropy < 1e-5:
            temperature = 0.0
        else:
            temperature = self._max_temperature * (self.N ** (self.theta / entropy))
        return temperature
