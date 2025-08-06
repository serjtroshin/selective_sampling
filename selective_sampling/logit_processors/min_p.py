import torch
import torch.nn.functional as F


class MinPLogitsProcessor:
    def __init__(self, min_p: float):
        self.min_p = torch.tensor(min_p).unsqueeze(0)

    def __call__(self, input_ids, logits):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        self.min_p = self.min_p.to(logits.device)

        # Move self.min_p to the same device as logits
        self.min_p = self.min_p.to(logits.device)

        # Calculate the probability of top-1 token
        argmax = logits.argmax(dim=-1)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        top_1_probs = probs.gather(1, argmax.unsqueeze(1))

        # Create a mask where probabilities are below the threshold
        mask = probs < self.min_p * top_1_probs

        # Set logits for those positions to -inf so they won't be sampled
        logits[mask] = float("-inf")
        return logits


if __name__ == "__main__":
    # test min_p
    logits = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
    min_p = 0.1
    min_p_processor = MinPLogitsProcessor(min_p)
    print(min_p_processor(None, logits))

    logits = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=torch.float32)
    min_p = 0.1
    min_p_processor = MinPLogitsProcessor(min_p)
    print(min_p_processor(None, logits))
