from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import logging

logging.basicConfig(level=logging.INFO)


import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MyCausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    dyntemp_logits: torch.FloatTensor = None


class MyCustomLlama(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        tokenizer,
        experiment_name=None,
        loss_type="bce",
        pos_weight=None,
    ):
        super().__init__(config)
        # You can initialize any additional layers or attributes here.

        self.pad_tok_idx = tokenizer.pad_token_id

        # introduce a linear classifier here:
        # last hidden state -> linear layer

        # all hidden states
        self.experiment_name = experiment_name
        if "all_hiddens" in self.experiment_name:
            self.classifier = [
                nn.Linear(config.hidden_size, 1)
                for _ in range(config.num_hidden_layers + 1)
            ]
            self.classifier = nn.ModuleList(self.classifier)

        elif (
            "last_hiddens" in self.experiment_name
            or "first_hiddens" in self.experiment_name
            or "middle_hiddens" in self.experiment_name
        ):
            self.classifier = nn.Linear(
                config.hidden_size, 1
            )  # only use the embedding of the top_1 token
        elif "embeddings" in self.experiment_name:
            raise NotImplementedError("Embedding classifier not implemented yet")
        else:
            raise ValueError(f"Invalid experiment name: {self.experiment_name}")

        # self.embedding_classifier = nn.Linear(config.hidden_size, 1)

        # freeze all parameters except self.classifier weights
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

        if loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            logging.info("Using BCE loss")
        elif loss_type == "bce_balanced":
            assert pos_weight is not None
            # if numpy float*, convert to tensor
            pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
            logging.info("Using balanced BCE loss with pos_weight: %f", pos_weight)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
            logging.info("Using MSE loss")
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

    def predict_all_hiddens(self, outputs, prompt_ids, input_ids):
        """
        Use all layers for prediction
        """
        assert isinstance(self.classifier, nn.ModuleList)
        assert len(outputs.hidden_states) == len(
            self.classifier
        ), f"{len(outputs.hidden_states)} != {len(self.classifier)}"
        logits = []
        for layer_id, hidden_states in enumerate(outputs.hidden_states):
            # We only want the hidden states corresponding to the 'input_ids' portion
            # prompt_ids.shape[1] = prompt_seq_len
            prompt_seq_len = prompt_ids.size(1)
            input_hidden_states = hidden_states[
                :, prompt_seq_len - 1 : -1, :
            ]  # shift to the left by one!

            assert (
                input_hidden_states.shape[1] == input_ids.shape[1]
            ), f"{input_hidden_states.shape[1]} != {input_ids.shape[1]}"

            # Pass the input hidden states through our classifier
            # Resulting shape: (batch_size, input_seq_len, 1)
            logits.append(self.classifier[layer_id](input_hidden_states))

        # Average the logits across layers
        logits = torch.stack(logits, dim=-1).mean(dim=-1)

        # Squeeze out the last dimension so we have (batch_size, input_seq_len)
        prediction = logits.squeeze(-1)

        return prediction

    def predict_single_hiddens(
        self, outputs, prompt_ids, input_ids, hiddens_index="last"
    ):
        """
        Using only the last hidden state for prediction
        """
        assert isinstance(self.classifier, nn.Linear)
        if hiddens_index == "last":
            last_hidden_states = outputs.hidden_states[-1]
        elif hiddens_index == "first":
            last_hidden_states = outputs.hidden_states[0]
        elif hiddens_index == "middle":
            last_hidden_states = outputs.hidden_states[len(outputs.hidden_states) // 2]
        else:
            raise ValueError(f"Invalid hiddens_index: {hiddens_index}")

        prompt_seq_len = prompt_ids.size(1)
        input_hidden_states = last_hidden_states[
            :, prompt_seq_len - 1 : -1, :
        ]  # shift to the left by one!

        assert (
            input_hidden_states.shape[1] == input_ids.shape[1]
        ), f"{input_hidden_states.shape[1]} != {input_ids.shape[1]}"
        logits = self.classifier(input_hidden_states)

        return logits.squeeze(-1)

    def classifier_train(self, prompt_ids, input_ids, labels, **kwargs):
        """
        prompt_ids: (batch_size, prompt_seq_len)
        input_ids:  (batch_size, input_seq_len)
        labels:     (batch_size, input_seq_len)  # binary labels for each input token
        """
        # Create attention masks
        prompt_attention_mask = (prompt_ids != self.pad_tok_idx).float()
        input_attention_mask = (input_ids != self.pad_tok_idx).float()

        # Concatenate prompt and input
        full_attention_mask = torch.cat(
            (prompt_attention_mask, input_attention_mask), dim=1
        )
        full_input = torch.cat((prompt_ids, input_ids), dim=1)

        # Forward through the base Llama model
        outputs = super().forward(
            input_ids=full_input,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # access prediction
        if "all_hiddens" in self.experiment_name:
            prediction = self.predict_all_hiddens(outputs, prompt_ids, input_ids)
        elif "last_hiddens" in self.experiment_name:
            prediction = self.predict_single_hiddens(
                outputs, prompt_ids, input_ids, hiddens_index="last"
            )
        elif "first_hiddens" in self.experiment_name:
            prediction = self.predict_single_hiddens(
                outputs, prompt_ids, input_ids, hiddens_index="first"
            )
        elif "middle_hiddens" in self.experiment_name:
            prediction = self.predict_single_hiddens(
                outputs, prompt_ids, input_ids, hiddens_index="middle"
            )
        else:
            raise ValueError(f"Invalid experiment name: {self.experiment_name}")
        assert prediction.shape == labels.shape, f"{prediction.shape} != {labels.shape}"

        # calculate loss
        loss = self.loss_fn(prediction, labels.float())
        assert loss.shape == prediction.shape, f"{loss.shape} != {prediction.shape}"
        # Mask out the padding positions in the input_ids
        # (batch_size, input_seq_len)
        loss = loss * input_attention_mask

        # Now aggregate the loss across tokens (e.g., mean)
        final_loss = loss.mean()

        return MyCausalLMOutputWithPast(
            loss=final_loss,
            # logits=outputs.logits,  # <--- store your predictions here
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            dyntemp_logits=prediction,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        num_logits_to_keep: int = 0,
        prompt_ids=None,  # special argument for training of classifier
        **loss_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for the model.
        """
        if prompt_ids is not None:
            # return self.classifier_train(
            if "embeddings" in self.experiment_name:
                return self.embedding_classifier_train(
                    prompt_ids=prompt_ids,
                    input_ids=input_ids,
                    labels=labels,
                    **loss_kwargs,
                )
            elif (
                "all_hiddens" in self.experiment_name
                or "last_hiddens" in self.experiment_name
                or "first_hiddens" in self.experiment_name
                or "middle_hiddens" in self.experiment_name
            ):
                return self.classifier_train(
                    prompt_ids=prompt_ids,
                    input_ids=input_ids,
                    labels=labels,
                    **loss_kwargs,
                )
            else:
                raise ValueError(f"Invalid experiment name: {self.experiment_name}")

        # Normal call to the base Llama model + classifier
        outputs: MyCausalLMOutputWithPast = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # this should be True
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        assert len(outputs.hidden_states) == len(
            self.classifier
        ), f"{len(outputs.hidden_states)} != {len(self.classifier)}"
        logits = 0.0
        for layer_id, hidden_states in enumerate(outputs.hidden_states):
            input_hidden_states = hidden_states[:, -1:, :]
            logits += self.classifier[layer_id](input_hidden_states)

        logits /= len(outputs.hidden_states)

        # Squeeze out the last dimension so we have (batch_size, input_seq_len)
        prediction = logits.squeeze(-1)

        outputs.dyntemp_logits = prediction

        return outputs


# Example usage:
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    embedding_dim = 8

    model = LinearConvolutionClassifier(embedding_dim)
    inputs = torch.randn(batch_size, seq_len, embedding_dim)
    outputs = model(inputs)
    print("Input shape:", inputs.shape)  # (2, 10, 8)
    print("Output shape:", outputs.shape)  # (2, 10, 1)
