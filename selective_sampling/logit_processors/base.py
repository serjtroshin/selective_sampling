from transformers import LogitsProcessorList
from abc import abstractmethod

class BaseLogitsProcessor:
    @classmethod
    @abstractmethod
    def from_kwargs(cls, **kwargs) -> LogitsProcessorList:
        pass