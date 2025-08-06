# add dummy logit processor
from transformers import LogitsProcessorList, LogitsProcessor
from selective_sampling.logit_processors.base import BaseLogitsProcessor


class DummyLogitProcessor(LogitsProcessor):
    def __init__(self, *args, **kwargs):
        # you can store the parameters here e.g. guidence coefficient
        pass

    def __call__(self, input_ids, logits):
        return logits


class DummyLogitsProcessorList(BaseLogitsProcessor):
    @classmethod
    def from_kwargs(cls, kwargs) -> LogitsProcessorList:
        return []
