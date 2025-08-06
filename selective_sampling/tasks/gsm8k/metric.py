import logging
import pickle
from typing import List
import numpy as np
import re
import string
from lm_eval.api.registry import register_aggregation, register_metric

eval_logger = logging.getLogger("lm-eval")



def exact_match_hf_evaluate(
    predictions: List[List[str]],
    references: List[str],
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    # references should be List[str] but it gets splitted
    # pred: [['18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18', '18']]
    # ref: [['1', '8']]
    # pickle.dump(predictions, open("predictions.pkl", "wb"))
    # pickle.dump(references, open("references.pkl", "wb"))
    
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([[re.sub(s, "", x) for x in prediction_list] for prediction_list in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references[None, :]

    # micro_aggregation
    score_max = np.max(score_list, axis=1)
    score_min = np.min(score_list, axis=1)
    score_avg = np.mean(score_list, axis=1)
    score_std = np.std(score_list, axis=1)

    return {"exact_match_max": np.mean(score_max), "exact_match_min": np.mean(score_min), "exact_match_avg": np.mean(score_avg), "exact_match_std": np.mean(score_std)}


@register_metric(
    metric="exact_match_avg",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def sample_exact_match(**kwargs):
    return exact_match_hf_evaluate(**kwargs)


if __name__ == "__main__":
    # write test
    # prediction_lists = [["aa", "b", "c"], ["a", "b", "c"]]
    # references = ["a", "b", "c"]
    # print(exact_match_hf_evaluate(prediction_lists, references))

    predictions = pickle.load(open("predictions_1.pkl", "rb"))
    references = pickle.load(open("references_1.pkl", "rb"))
    print(predictions)
    print(references)
    print(exact_match_hf_evaluate(predictions, references))
