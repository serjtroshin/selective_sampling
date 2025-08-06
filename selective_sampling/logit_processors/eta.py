import torch
import transformers
import random

device = 'cuda'

class EtaLogitsProcessor:
  """Our proposed eta sampling warper."""
  def __init__(self, eta):
    self.eta = eta
    self.filter_value = -float("Inf")

  def __call__(self, input_ids, logits):
    probabilities = logits.softmax(dim=-1)
    entropy = torch.distributions.Categorical(probs=(logits).softmax(dim=-1)).entropy()
    eta = min(self.eta, torch.sqrt(torch.tensor(self.eta))*torch.exp(-entropy))
    indices_to_remove = probabilities < eta
    max_word = torch.argmax(logits,dim=-1)
    indices_to_remove[...,max_word.squeeze()] = 0
    new_scores = logits.masked_fill(indices_to_remove, self.filter_value)
    return new_scores