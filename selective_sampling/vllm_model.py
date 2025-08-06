from copy import copy, deepcopy
import os
import pickle
from tqdm import tqdm
from transformers import GenerationMixin
from typing import Optional, Tuple, Union

import torch
import transformers

from lm_eval import utils
from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.models.utils import (
    Collator,
    stop_sequences_criteria,
)
from lm_eval.api.instance import Instance
from typing import List
from vllm import RequestOutput, SamplingParams, CompletionOutput
from vllm.config import CacheConfig, VllmConfig

from selective_sampling.utils import (
    parse_logit_processor_kwargs,
    warn_about_new_args,
    parse_sampling_config,
)

import logging

logging.basicConfig(level=logging.INFO)


from selective_sampling.utils import load_config, CONFIGS
from selective_sampling.temperature_classifier.utils import (
    parse_token_importance_classifier_sampling,
    parse_temperature_classifier_sampling,
    TokenImportanceConfig,
    TokenImportanceLogitProcessor,
    TemperatureClassifier,
    TemperatureClassifierLogitProcessor,
)

from selective_sampling.logit_processors import (
    TopPLogitsProcessor,
    MinPLogitsProcessor,
    DoubleTemperatureMinPLogitsProcessor,
    EpsilonLogitsProcessor,
    EtaLogitsProcessor,
)


GLOBAL_I = 0


@register_model("vllm_wrapper")
class MyVLLMModelWrapper(VLLM):
    def __init__(self, pretrained, *args, **kwargs):
        kwargs, logit_processor_list = parse_logit_processor_kwargs(kwargs)
        # Config for dataset creation
        # provide: token_importance
        self.diversity_classifier_sampling: TokenImportanceConfig = (
            parse_token_importance_classifier_sampling(kwargs)
        )
        # test logit processor
        if self.diversity_classifier_sampling is not None:
            self._tok_importance_logit_processor = TokenImportanceLogitProcessor(
                token_importance_config=self.diversity_classifier_sampling,
            )

        # provide: temperature_classifier_configs: "temperature_classifier_configs.yaml"
        self.temperature_classifier: TemperatureClassifier = (
            parse_temperature_classifier_sampling(kwargs)
        )

        kwargs["enforce_eager"] = True

        super().__init__(pretrained, *args, **kwargs)

        self.logits_processor_list = logit_processor_list

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        print(kwargs)
        print("-------- <", flush=True)
        kwargs = self._parse_sampling_config(kwargs)
        print(kwargs)
        assert (
            "seed" in kwargs
        ), "seed is required for sampling, 1 seed is used to sample 1 sample for each prompt"

        # assert all requests are different
        # assert len(set(map(tuple, requests))) == len(requests)

        logits_processor_list = deepcopy(self.logits_processor_list)
        if self.diversity_classifier_sampling:
            # generate data for temperature classifier
            logit_processor = TokenImportanceLogitProcessor(
                token_importance_config=self.diversity_classifier_sampling,
                greedy_samples=self._tok_importance_logit_processor.greedy_samples,
            )
            self.logits_log = logit_processor.log
            logits_processor_list = logits_processor_list + [logit_processor]
        else:
            self.logits_log = []

        if self.temperature_classifier:
            """
            Important: we need to ensure we first do top_p e.t.c. and then change temperature
            One solution: use logits top_p processor
            """
            print(
                "Dynamically setting temperature: {}".format(
                    self.temperature_classifier.config
                )
            )

            top_p = 1.0
            min_p = 0.0
            epsilon = 0.0
            eta = 0.0

            if "top_p" in kwargs:
                top_p = kwargs.pop("top_p")

            if "min_p" in kwargs:
                min_p = kwargs.pop("min_p")

            if "epsilon" in kwargs:
                epsilon = kwargs.pop("epsilon")

            if "eta" in kwargs:
                eta = kwargs.pop("eta")

            if "support_temperature" in kwargs:
                # use double temperature sampling
                temperature = kwargs.pop(
                    "temperature", 1.0
                )  # sampling temperature (after crop)
                support_temperature = kwargs.pop(
                    "support_temperature"
                )  # temperature for the support size
                print(
                    "Using DoubleTemperatureMinPLogitsProcessor with support temperature={}".format(
                        support_temperature
                    )
                )
                min_p_processor = DoubleTemperatureMinPLogitsProcessor(
                    min_p, temperature, support_temperature
                )
                logits_processor_list = logits_processor_list + [min_p_processor]
                # set kwargs temperature to 1.0
                kwargs["temperature"] = 1.0
            else:
                # use default regime for temperature sampling
                print("Temperature will be hangled in logit processor")
                logit_processor = TemperatureClassifierLogitProcessor(
                    temperature_classifier=self.temperature_classifier,
                    temperature=kwargs.get("temperature", 1.0),
                )
                logits_processor_list = logits_processor_list + [logit_processor]

                print(self.temperature_classifier)

                if top_p != 1.0:
                    top_p_processor = TopPLogitsProcessor(top_p)
                    logits_processor_list = logits_processor_list + [top_p_processor]

                if min_p != 0.0:
                    min_p_processor = MinPLogitsProcessor(min_p)
                    logits_processor_list = logits_processor_list + [min_p_processor]

                if epsilon != 0.0:
                    epsilon_processor = EpsilonLogitsProcessor(epsilon)
                    logits_processor_list = logits_processor_list + [epsilon_processor]

                if eta != 0.0:
                    eta_processor = EtaLogitsProcessor(eta)
                    logits_processor_list = logits_processor_list + [eta_processor]

                # set kwargs temperature to 1.0
                kwargs["temperature"] = 1.0

                assert (
                    kwargs["temperature"] == 1.0
                ), "Temperature should be 1.0 in kwargs: it is processed in logit processor"
                # assert no min_p or top_p in kwargs
                assert "top_p" not in kwargs, "top_p should not be in kwargs"
                assert "min_p" not in kwargs, "min_p should not be in kwargs"
                assert "epsilon" not in kwargs, "epsilon should not be in kwargs"
                assert "eta" not in kwargs, "eta should not be in kwargs"

        logging.info("logits_processor_list: {}".format(logits_processor_list))

        if generate:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                stop=stop,
                logits_processors=logits_processor_list,
                logprobs=True,
                **kwargs,
            )

        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )

        output = self.model.generate(
            prompt_token_ids=requests,
            sampling_params=sampling_params,
            use_tqdm=True if self.batch_size == "auto" else False,
            lora_request=self.lora_request,
        )

        return output

    def _parse_sampling_config(self, kwargs: dict) -> dict:
        sampling_kwargs = parse_sampling_config(kwargs)

        if "do_sample" in sampling_kwargs:
            del sampling_kwargs["do_sample"]
        if (
            "num_beams" in sampling_kwargs
        ):  # should not use config name to determine if beam search is on
            # beam search mode is on
            del sampling_kwargs["early_stopping"]  # why deleting?
            num_beams = sampling_kwargs.pop("num_beams", 1)
            sampling_kwargs["best_of"] = num_beams

        return sampling_kwargs

    # I will add it here, because I need to modify the example saving code to save extra information! But other logic won't change
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        requests_ = requests
        # batch tokenize contexts
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            # add EOS token to stop sequences
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            print("max_gen_toks", max_gen_toks)
            print("max_length", self.max_length)
            # set the max length in tokens of inputs ("context_enc")
            # max len for inputs = max length, minus room to generate the max new tokens
            max_ctx_len = self.max_length - max_gen_toks
            context_encoding = [x[-max_ctx_len:] for x in context_encoding]

            # perform batched generation
            cont = self._model_generate(
                requests=context_encoding,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs,
            )
            # with open(f"model_generate_output.pkl", "wb") as f:
            #     pickle.dump(cont, f)

            # cache generations
            for output, context in zip(cont, context):

                completion_output: CompletionOutput = output.outputs[0]
                generated_text = completion_output.text

                logprobs = completion_output.logprobs

                # format of logprobs: logprobs=[{1271: Logprob(logprob=-0.8569104671478271, rank=1, decoded_token='To')}, {1505: Logprob(logprob=-0.9422308206558228, rank=1, decoded_token=' find')}, {279: Logprob(logprob=-0.0912923738360405, rank=1, decoded_token=' the')}, {4876: Logprob(logprob=-0.010958724655210972, rank=1, decoded_token=' graph')}, {315: Logprob(logprob=-0.0012287693098187447, rank=1, decoded_token=' of')},
                # transform logprobs to a list of logprobs for rank=1 tokens. we need to first filter the logprobs by rank=1
                def take_rank_1(logprob_dict) -> float | None:
                    for logprob in logprob_dict.values():
                        if logprob.rank == 1:
                            return logprob.logprob
                    return None

                if logprobs is not None:
                    logprobs = [take_rank_1(logprob_dict) for logprob_dict in logprobs]

                # for token_id, log_prob in zip(completion_output.token_ids, logprobs):
                #     print(f"Token: {self.tokenizer.decode(token_id)} | Logprob: {log_prob}")
                # input()
                prompt_token_str = context
                # detokenize the prompt
                prompt_token_ids = self.tok_encode(
                    prompt_token_str, add_special_tokens=self.add_bos_token
                )

                logits = []
                top_token_ids = []
                if len(self.logits_log) > 0:
                    logging.info("parsing logits for greedy sequence")
                    for position in range(len(completion_output.token_ids)):
                        catted = tuple(
                            list(prompt_token_ids)
                            + list(completion_output.token_ids[:position])
                        )
                        assert (
                            catted in self.logits_log
                        ), "Prompt not found in logits log"
                        logits.append(self.logits_log[catted]["top_k_logits"].tolist())
                        top_token_ids.append(
                            self.logits_log[catted]["top_k_token_ids"].tolist()
                        )
                    assert len(logits) == len(
                        completion_output.token_ids
                    ), "Logits should be the same length as the generated sequence"

                result_to_append = (
                    generated_text,
                    {
                        "token_ids": completion_output.token_ids,
                        "prompt_token_ids": prompt_token_ids,
                        "logprobs": logprobs,
                        "top_logits": logits,
                        "top_tokens": top_token_ids,
                    },
                )
                res.append(result_to_append)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        reordered = re_ords.get_original(res)
        # with open(f"reordered.pkl", "wb") as f:
        #     pickle.dump([reordered, requests_], f)

        new_reord = []
        for reord, req in zip(reordered, requests_):
            req.extra = reord[1]
            new_reord.append(reord[0])  # remove extra info

        return new_reord
