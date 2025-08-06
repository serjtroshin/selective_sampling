import re
from typing import Any, List
from collections import Counter


def allclose(a, b):
    return abs(a - b) < 1e-6


class GSM8KFilter:
    @classmethod
    def is_sample_correct(cls, sample):
        return allclose(sample["exact_match"], 1.0)

    @classmethod
    def is_sample_parsable(cls, sample):
        return sample["filtered_resps"][0] == "[invalid]"


class MajorityVote:
    @classmethod
    def aggregate(cls, samples: List[Any], invalid_mark: str) -> Any:
        filtered_resps = [sample["filtered_resps"][0] for sample in samples]
        # filter out invalid responses
        filtered_resps = [resp for resp in filtered_resps if resp != invalid_mark]
        if len(filtered_resps) == 0:
            return {
                "exact_match": 0.0,
                "filtered_resps": filtered_resps,
                "normalized_resps": [],
                "majority": invalid_mark,
                "correct_answer": samples[0]["target"],
            }
        normalized_resps = [cls.normalize_response(resp) for resp in filtered_resps]
        # majority voting
        counter = Counter(normalized_resps)
        majority = counter.most_common(1)[0][0]
        correct_answer = samples[0]["target"]
        return {
            "exact_match": 1.0 if majority == correct_answer else 0.0,
            "filtered_resps": filtered_resps,
            "normalized_resps": normalized_resps,
            "majority": majority,
            "correct_answer": correct_answer,
        }

    @classmethod
    def normalize_response(cls, response: str) -> str:
        regexes_to_ignore = [r",", r"\$", r"(?s).*####", r"\.$"]
        for s in regexes_to_ignore:
            response = re.sub(s, "", response)
        return response


class MinervaFilter:
    @classmethod
    def is_sample_correct(cls, sample):
        return allclose(sample["exact_match"], 1.0)

    @classmethod
    def is_sample_parsable(cls, sample):
        raise NotImplementedError


class MMLUFilter:
    @classmethod
    def is_sample_correct(cls, sample):
        return allclose(sample["exact_match"], 1.0)

    @classmethod
    def is_sample_parsable(cls, sample):
        return sample["filtered_resps"][0] == "[invalid]"


def task_name_to_filter(task_name):
    if "gsm8k" in task_name or "gpqa" in task_name:
        return GSM8KFilter
    if "minerva" in task_name:
        return MinervaFilter  # todo edit
    if "mmlu" in task_name:
        return MMLUFilter
    else:
        raise NotImplementedError()


def task_name_to_aggregator(task_name):
    if task_name.startswith("gsm8k"):
        return MajorityVote
    else:
        raise NotImplementedError()


def is_sample_correct(sample, task_name):
    filter = task_name_to_filter(task_name)
    return filter.is_sample_correct(sample)


def is_sample_parsable(sample, task_name):
    filter = task_name_to_filter(task_name)
    return filter.is_sample_parsable(sample)


def aggregate_samples(samples, task_name, invalid_mark="[invalid]"):
    aggregator = task_name_to_aggregator(task_name)
    return aggregator.aggregate(samples, invalid_mark=invalid_mark)
