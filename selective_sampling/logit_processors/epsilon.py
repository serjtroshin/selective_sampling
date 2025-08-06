import torch
import transformers
import random

device = 'cuda'

class EpsilonLogitsProcessor:
  """
  [`LogitsProcessor`] that performs epsilon, i.e. restricting to tokens with absolute prob > prob_cut_off.
  Takes single argmax token if no tokens satisfy this constraint.
  Args:
      epsilon (`float`):
          If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
      filter_value (`float`, *optional*, defaults to `-float("Inf")`):
          All filtered values will be set to this float value.
  """
  def __init__(self, epsilon):
    self.epsilon = epsilon
    self.filter_value = -float("Inf")

  def __call__(self, input_ids, logits):
    probabilities = logits.softmax(dim=-1)
    indices_to_remove = probabilities < self.epsilon
    max_word = torch.argmax(logits,dim=-1)
    indices_to_remove[...,max_word.squeeze()] = 0
    new_scores = logits.masked_fill(indices_to_remove, self.filter_value)
    return new_scores
