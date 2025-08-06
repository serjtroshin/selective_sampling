import argparse
import logging
from pathlib import Path
from typing import List
from selective_sampling.subprocess_utils import handle_subprocess
from selective_sampling.diversity_utils import process_json, get_script

SUPPORTED_TASKS = {
    "gdg_gsm8k_cot_llama": ("gdg_gsm8k_cot_llama", ["exact_match,flexible-extract"]),
    "gsm8k_zeroshot": (
        "gdg_gsm8k_cot_llama_zeroshot",
        ["exact_match,flexible-extract"],
    ),
    "minerva_math_intermediate_algebra": (
        "gdg_minerva_math_intermediate_algebra",
        ["exact_match,none"],
    ),
    "minerva_math_prealgebra": ("gdg_minerva_math_prealgebra", ["exact_match,none"]),
    "symbolic_gsm8k_cot_llama_main": (
        "symbolic_gsm8k_cot_llama_main",
        ["exact_match,flexible-extract"],
    ),
    "symbolic_gsm8k_cot_llama_p1": (
        "symbolic_gsm8k_cot_llama_p1",
        ["exact_match,flexible-extract"],
    ),
    "symbolic_gsm8k_cot_llama_p2": (
        "symbolic_gsm8k_cot_llama_p2",
        ["exact_match,flexible-extract"],
    ),
    "mmlu_pro_social_sciences": (
        "mmlu_pro_social_sciences",
        ["exact_match,custom-extract"],
    ),
}

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter",
        default="correct",
        help="Filter to apply to the samples",
        choices=["correct", "parsable"],
    )
    parser.add_argument(
        "--override", action="store_true", help="Override existing files"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name",
        choices=SUPPORTED_TASKS,
    )
    parser.add_argument(
        "--override_task_metric",
        default=None,
        help="e.g. exact_match or factKB will be parsed from the folder",
    )
    parser.add_argument(
        "--parse_task_metric_from_model_name",
        action="store_true",
        help="Parse task metric from model name: used for dynamic temperature models: e.g. factKB or bertscore_f1",
    )

    parser.add_argument(
        "--samples_dir",
        type=str,
        default="./samples",
        help="Directory containing samples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=25,
        help="Number of samples per prompt : use other 5 samples if failed",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of prompts to process"
    )
    parser.add_argument(
        "--model",
        help="Model name",
        default=None,
    )
    parser.add_argument(
        "--model_from_seed_dir",
        help="determine model names from seed directory",
        default=None,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Metrics to run",
        default="AveragedDistinctNgrams",
        choices=["AveragedDistinctNgrams", "SentBert", "LevenshteinDistance"],
    )  # SentBert,
    parser.add_argument(
        "--python_path",
        type=str,
        help="Path to the python executable",
        default="python",
    )
    parser.add_argument("--mode", choices=["sbatch", "bash"], default="bash")

    return parser.parse_args()


def model_list_from_seed_dir(seed_dir) -> List[str]:
    models = []
    logging.info(f"Searching for models in {seed_dir}")
    for path in Path(seed_dir).rglob("results_*.json"):
        # 1. find part with seed: seed_*
        seed_i = None
        for i in range(len(path.parts)):
            if path.parts[i].startswith("seed_"):
                seed_i = i
                break
        # 2. extract model, method, sampling_param, temp_param
        model = path.parts[seed_i + 1 : -1]
        model_name = "/".join(model)
        models.append(model_name)
    assert len(models) > 0
    return models


def parse_metric_from_model_name(model):
    raise NotImplementedError


def main():
    args = parse_args()

    if args.model_from_seed_dir is not None:
        assert args.model is None
        models = list(model_list_from_seed_dir(args.model_from_seed_dir))
        print(f"Found models: {models}")
        tasks = []
        for model in models:
            args.model = model

            if args.parse_task_metric_from_model_name:
                logging.info(f"parsing metric from model name: {model}")
                args.override_task_metric = parse_metric_from_model_name(model)

            tasks.extend(process_json(args))
    else:
        tasks = process_json(args)

    if not args.override:
        new_tasks = []
        # find completed tasks
        for task in tasks:
            result_dir = task.replace("data/raw/", "data/with_metrics/")
            if Path(result_dir).exists():
                print(f"Skipping {task} as it already exists")
            else:
                new_tasks.append(task)
        tasks = new_tasks

    print("TASKS:")
    for task in tasks:
        print(task)

    # input("Press Enter to continue...")

    with open("outputs/diversity_log.txt", "w") as f:
        for task in tasks:
            f.write(task)
            f.write("\n")

    for task in tasks:
        sbatch_script = get_script(args.mode, args.python_path, task, args.metrics)
        handle_subprocess(sbatch_script.split())


if __name__ == "__main__":
    main()
