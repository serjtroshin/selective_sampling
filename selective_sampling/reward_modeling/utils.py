import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch.utils.data import ConcatDataset, Subset
from distutils.util import strtobool

from tqdm import tqdm
import yaml

from datasets import load_dataset
from transformers import AutoTokenizer

from datasets import load_dataset, Features, Value, Sequence, Dataset
from embedding_model import EmbeddingModel

from wrapper_model import MyCustomLlama
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
)

logging.basicConfig(level=logging.INFO)


class BinarizationStrategy:
    def __init__(self, strategy: str):
        self.strategy = strategy
        assert self.strategy in ["default", "none"]

    def __str__(self):
        return self.strategy

    def __repr__(self):
        return self.strategy

    def binarize_reward(self, reward: float, epislon: float) -> float | int:
        if self.strategy == "none":
            return reward
        elif self.strategy == "default":
            if reward > 1.0 - epislon:
                return 1
            else:
                return 0
        else:
            raise ValueError(f"Unknown binarization strategy: {self.strategy}")


def task_2_subtask(task_name):
    if "minerva" in task_name:
        return ""
    else:
        return "exact_match"


def balance_datasets(train_datasets):
    """Upsample underrepresented datasets to balance training data."""

    dataset_sizes = [len(d) for d in train_datasets]
    max_size = max(dataset_sizes)  # Find the largest dataset size

    balanced_datasets = []

    for d in train_datasets:
        size = len(d)
        if size < max_size:
            # Convert indices to a list of Python ints to avoid numpy.int64 issues
            indices = [
                int(i) for i in np.random.choice(range(size), max_size, replace=True)
            ]
            balanced_datasets.append(Subset(d, indices))
        else:
            balanced_datasets.append(d)

    return ConcatDataset(balanced_datasets)


def task_2_max_gen_tokens(dataset_name):
    if "gpqa" in dataset_name:
        return "1024"
    else:
        return "512"


def get_dataset(
    args,
    tokenizer=None,
    same_tokenizer=False,
) -> tuple[ConcatDataset, dict[str, Subset]]:
    """
    return concatenated datasets: train, evals
    """

    print("args", args)

    train_datasets, evals = [], {}

    task_dir = args.task_dir
    print("task_dir", task_dir)
    print("args.tasks", args.tasks)
    for data_config in args.tasks:
        dataset_name = data_config
        data_dir = task_dir.replace(r"{{task}}", dataset_name).replace(
            r"{{max_gen_tokens}}", task_2_max_gen_tokens(dataset_name)
        )
        if r"{{subtask}}" in data_dir:
            data_dir = data_dir.replace(r"{{subtask}}", task_2_subtask(dataset_name))
            print(">> data_dir for task", dataset_name, ":", data_dir)
        if same_tokenizer:
            train, val = get_one_dataset_same_tokenizer(
                args,
                dataset_name,
                data_dir,
                tokenizer,
                do_binarization=BinarizationStrategy(args.dataset_binarization),
            )
        train_datasets.append(train)

        if val is not None:
            evals[dataset_name] = val

    if args.task_balance:
        print("train_datasets", train_datasets)
        train = balance_datasets(train_datasets)

    else:
        train = ConcatDataset(train_datasets)

    print("Dataset stats before sampling:")
    total = len(train)
    for d in train.datasets:
        if isinstance(d, Subset):
            name = f"Subset of {type(d.dataset).__name__}"
            if hasattr(d.dataset, "name"):
                name += f" ({d.dataset.name})"
        else:
            name = type(d).__name__
            if hasattr(d, "name"):
                name += f" ({d.name})"
        print(f"{name}: {len(d)} ({len(d) / total:%})")
    print(f"Total train: {total}")

    print("Dataset stats for eval:")
    for dataset_name, val in evals.items():
        print(f"{dataset_name}: {len(val)}")

    return train, evals


def load_json_dataset(
    data_dir, epsilon: float, do_binarization: BinarizationStrategy
) -> List[Dict[str, Any]]:
    """
    read the json list(dict) dataset: token importance dataset
    format:
     doc_id:
        [{'text': 'We', 'reward': 0.8881743550300598, 'token_id': 1687, 'position': 0},...]
    returns:
    dataset: list of dict
        {'doc_id': 'doc_id', 'token_ids': [...], 'risks': [0.22, ...]}


    binarizes rewards with epsilon
    """

    with open(data_dir) as f:
        dataset = json.loads(f.read())

    print("loaded {} samples".format(len(dataset)))

    # combine the jsons together:
    output = []
    for doc_id in dataset:
        sample = dataset[doc_id]

        cur_json = {
            "doc_id": doc_id,
            "prompt_ids": sample["prompt_token_ids"],
        }
        cur_json["token_ids"] = [
            tok_info["token_id"] for tok_info in sample["token_info"]
        ]
        cur_json["risks"] = [
            1
            - do_binarization.binarize_reward(
                tok_info["reward"], epsilon
            )  # 1.0 means risky, 0.0 means safe
            for tok_info in sample["token_info"]
        ]
        output.append(cur_json)

    return output


def get_one_dataset_same_tokenizer(
    args,
    dataset_name,
    data_dir,
    tokenizer,
    do_binarization: BinarizationStrategy,
):
    """
    When using the same model to generate data and to train, we can use the same tokenizer
    """
    print("loading dataset", dataset_name, "from", data_dir)
    # load dataset from jsonl
    dataset = load_json_dataset(
        data_dir, args.reward_epsilon, do_binarization=do_binarization
    )

    # filter out sample where the greedy answer is incorrect
    # dataset = [data for data in dataset if is_sample_correct(data)]
    # print("leaving only correct samples. N samples after filtering: ", len(dataset))
    # print(dataset[0])

    # convert to dataset
    dataset = Dataset.from_generator(dataset.__iter__)
    print("loaded {} samples".format(len(dataset)))

    dataset = dataset.rename_columns(
        {"token_ids": "input_ids", "risks": "labels", "prompt_ids": "prompt_ids"}
    )
    columns_to_keep = ["input_ids", "labels", "prompt_ids"]

    dataset = dataset.remove_columns(
        list(set(dataset.column_names) - set(columns_to_keep))
    )
    print("inspecting the dataset")
    print("processed dataset", dataset[0])
    all_labels: List[int] = [
        label for label_list in dataset["labels"] for label in label_list
    ]
    counts = np.unique(all_labels, return_counts=True)
    print("label distribution", counts)
    if args.class_balance:
        pos_weight = counts[1][0] / counts[1][1]  # negative / positive
        print("pos_weight", pos_weight)
        args.pos_weight = pos_weight
    else:
        args.pos_weight = None

    # split dataset by doc_id into train/val

    test_size = 100
    if len(dataset) < 500:
        test_size = 0.1

    dataset = dataset.train_test_split(test_size=test_size)
    train = dataset["train"]
    val = dataset["test"]
    assert "input_ids" in train.column_names
    assert "labels" in train.column_names

    return train, val


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # set pad idx
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name,
    tokenizer,
    evaluate=False,
    bf16=False,
    fp16=False,
    from_checkpoint=None,
    experiment_name=None,
    loss_type="bce",
    pos_weight=None,  # for class imbalance
):
    # Example: load a pretrained config from an existing model or local checkpoint
    torch_dtype = torch.float32
    if fp16:
        torch_dtype = torch.float16
    if bf16:
        torch_dtype = torch.bfloat16

    config = LlamaConfig.from_pretrained(model_name, torch_dtype=torch_dtype)

    # Load finetuned checkpoint
    if from_checkpoint:
        # try using from_pretrained

        custom_model = MyCustomLlama.from_pretrained(
            from_checkpoint,
            torch_dtype=torch_dtype,
            tokenizer=tokenizer,
            experiment_name=experiment_name,
        )

        # print("Loading from checkpoint: ", from_checkpoint)
        # with open(from_checkpoint / "model.safetensors.index.json", "r") as f:
        #     weight_map = json.loads(f.read())["weight_map"]

        # state_dict = load_shared_model(str(from_checkpoint), weight_map)
        # custom_model = MyCustomLlama(config, tokenizer)
        # custom_model.load_state_dict(state_dict)

        # eval mode
        custom_model.eval()
        logging.info(f"Loaded model from {from_checkpoint}")

        # map to cuda
        custom_model = custom_model.to("cuda")
        return custom_model

    custom_model = MyCustomLlama(
        config,
        tokenizer,
        experiment_name=experiment_name,
        loss_type=loss_type,
        pos_weight=pos_weight,
    )

    # Load pretrained state_dict
    pretrained_model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )
    custom_model.load_state_dict(pretrained_model.state_dict(), strict=False)
    del pretrained_model

    # map to cuda
    custom_model = custom_model.to("cuda")
    return custom_model


def load_embeddings_model(
    embeddings_path,
    tokenizer,
    experiment_name,
    loss_type,
    pos_weight,
    hidden_dim=32,
    n_layers=1,
    k_past_tokens=5,
    from_checkpoint=None,
):
    embeddings = torch.load(embeddings_path)
    model = EmbeddingModel(
        embeddings,
        tokenizer,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        k_past_tokens=k_past_tokens,
        pos_weight=pos_weight,
    )
    if from_checkpoint is not None:
        # load the weights
        print("loading the weights from", from_checkpoint)
        # load from from_checkpoint/model.safetensors.index.json
        state_dict = load_file(from_checkpoint / "model.safetensors")
        model.load_state_dict(state_dict)
        print("loaded the weights")

    model = model.to("cuda")
    return model


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # calculate max length
        max_length = max(len(feature["input_ids"]) for feature in features)

        if self.max_length is not None and self.max_length > 0:
            max_length = min(self.max_length, max_length)

        # pad input_ids: right side
        input_ids = [
            f["input_ids"]
            + [self.tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]
        batch = {"input_ids": torch.tensor(input_ids)}

        # pad prompt_ids: left side
        max_prompt_length = max(len(feature["prompt_ids"]) for feature in features)
        prompt_ids = [
            [self.tokenizer.pad_token_id] * (max_prompt_length - len(f["prompt_ids"]))
            + f["prompt_ids"]
            for f in features
        ]
        batch["prompt_ids"] = torch.tensor(prompt_ids)

        # pad label_ids
        label_ids = [
            f["labels"] + [-100] * (max_length - len(f["labels"])) for f in features
        ]
        batch["labels"] = torch.tensor(label_ids)

        return batch


def get_datacollator(tokenizer):
    default_data_collator = DataCollatorWithPadding(tokenizer)
    # test data collator
    return default_data_collator


def _strtobool(x):
    return bool(strtobool(x))


def read_yamls(dir):
    args = {}
    no_conf = True

    for config_file in Path(dir).glob("**/*.yaml"):
        no_conf = False
        with config_file.open("r") as f:
            args.update(yaml.safe_load(f))

    if no_conf:
        print(f"WARNING: No yaml files found in {dir}")

    return args
