from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MyCausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    dyntemp_logits: torch.FloatTensor = None


class LinearConvolutionClassifier(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim=32, k_past_tokens=5, do_conv_padding=True
    ):
        """
        Args:
            embedding_dim (int): Dimensionality of each token embedding.
        """
        super().__init__()
        # out_channels=1 for a single "score" per token.
        # If you need multiple classes per token, use out_channels=num_classes.
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            kernel_size=k_past_tokens,
            stride=1,
            padding=0,  # We'll do our own manual (left) padding below.
        )
        self.k_past_tokens = k_past_tokens
        self.do_conv_padding = do_conv_padding
        # self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch_size, seq_len, embedding_dim)

        Returns:
            Tensor: Shape (batch_size, seq_len, 1)
        """
        # x -> (batch_size, embedding_dim, seq_len)
        x = x.permute(0, 2, 1)

        # 1) Manually pad left with 4 zeros (kernel_size - 1) for "causal" style.
        #    This ensures index i in the output sees the previous 5 tokens (including itself).

        if self.do_conv_padding:
            x = F.pad(
                x, (self.k_past_tokens - 1, 0)
            )  # pads the last dimension: left=4, right=0

        # 2) Apply convolution
        x = self.conv(x)  # -> (batch_size, hidden_dim, seq_len)

        # 3) (Optional) Permute back to (batch_size, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)

        # # 4) Apply activation function (e.g., relu)
        # x = F.sigmoid(x)

        # # apply linear layer
        # x = self.linear(x)

        return x


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        embeddings,
        tokenizer,
        hidden_dim=32,
        n_layers=1,
        k_past_tokens=5,
        do_conv_padding=True,  # true for training but false for eval where we need only last prediction
        pos_weight=None,
    ):
        super(EmbeddingModel, self).__init__()
        self.embeddings: nn.Embedding = embeddings  # nn.Embedding
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.device = "cuda"

        self.embed_dim = self.embeddings.embedding_dim

        layers = [
            LinearConvolutionClassifier(
                self.embed_dim,
                hidden_dim=hidden_dim,
                k_past_tokens=k_past_tokens,
                do_conv_padding=do_conv_padding,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(*layers)

        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=torch.tensor(pos_weight, device=self.device),
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, prompt_ids, input_ids, labels, **kwargs):
        input_attention_mask = (input_ids != self.pad_token_id).float()

        input_hidden_states = self.embeddings(input_ids)
        logits = self.classifier(input_hidden_states)

        # Squeeze out the last dimension so we have (batch_size, input_seq_len)
        prediction = logits.squeeze(-1)

        if labels is not None:
            assert (
                prediction.shape == labels.shape
            ), f"{prediction.shape} != {labels.shape}"

            # calculate loss
            loss = self.loss_fn(prediction, labels.float())
            assert loss.shape == prediction.shape, f"{loss.shape} != {prediction.shape}"
            # Mask out the padding positions in the input_ids
            # (batch_size, input_seq_len)
            loss = loss * input_attention_mask

            # Now aggregate the loss across tokens (e.g., mean)
            final_loss = loss.mean()
        else:
            final_loss = None

        return MyCausalLMOutputWithPast(
            loss=final_loss,
            dyntemp_logits=prediction,
            # logits=prediction,
        )
