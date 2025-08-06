import argparse
import json
import csv
import logging
import os
from pathlib import Path
from typing import List

from multiprocessing import Pool
import warnings

# get current path
PATH = Path(__file__).resolve().parent.parent
# add to the path
import sys

sys.path.append(str(PATH))

PYTHON_GDG_PATH = os.getenv("PYTHON_GDG_PATH", "python")


from selective_sampling.task_utils import is_sample_correct, is_sample_parsable


def get_file_name(args):
    if args.filter != "":
        return f"diversity.{args.filter}_samples.csv"
    else:
        return "diversity.csv"


def get_metric_name(task_name, args):
    if args.override_task_metric is not None:
        return args.override_task_metric

    if task_name in [
        "gsm8k",
        "gdg_gsm8k_cot_llama",
        "minerva_math_intermediate_algebra",
        "minerva_math_prealgebra",
        "symbolic_gsm8k_cot_llama_main",
        "symbolic_gsm8k_cot_llama_p1",
        "symbolic_gsm8k_cot_llama_p2",
        "gpqa_main_cot_n_shot_v2",
        "mmlu_pro_social_sciences",
        "mmlu_pro",
    ]:
        return "exact_match"
    elif task_name == "gdg_ifeval":
        return "prompt_level_loose_acc"
    elif task_name == "gdg_xsum":
        return "factKB"
    else:
        raise ValueError(f"Task {task_name} not supported")


INCORRECT_RESPONSE_LABEL = "<<invalid_response>>"
# we will save a label instead of the response for incorrect samples


def quote_response(response: str):
    # replace all \n
    response = response.replace("\n", "\\n")
    return response


def process_temperature_file(
    temperature, temp_dir, num_samples, model, task_name, args
):
    document_ids_t = []
    responses = {}
    responses_metrics = {}
    is_error = False
    failed_samples = []
    for i in range(num_samples):
        seed_file = "seed_" + str(i)
        sample_dir = os.path.join(temp_dir, seed_file, model)
        if not os.path.isdir(sample_dir):
            is_error = True
            print(f"No samples found in {sample_dir} -> error")
            failed_samples.append(sample_dir)
            break
        for file in os.listdir(sample_dir):
            if file.startswith("samples"):
                with open(os.path.join(sample_dir, file)) as f:
                    for line in f:
                        data = json.loads(line)
                        document_id = str(data["doc_id"]) + "_" + str(temperature)
                        if document_id not in document_ids_t:
                            document_ids_t.append(document_id)
                        doc_id_seed = document_id + "_" + str(i)
                        response = data["resps"][0][0]
                        response_metrics = data[get_metric_name(task_name, args)]
                        if args.filter == "correct":
                            if not is_sample_correct(data, task_name):
                                response = INCORRECT_RESPONSE_LABEL  # mark as incorrect
                        if args.filter == "parsable":
                            if not is_sample_parsable(data, task_name):
                                response = INCORRECT_RESPONSE_LABEL
                        response = quote_response(response)
                        responses[doc_id_seed] = (
                            response  # if multiple responses for one doc_id, keep one (bug)
                        )
                        responses_metrics[doc_id_seed] = response_metrics
                        # else:
                        # responses[doc_id_seed].append(response)
    return document_ids_t, responses, responses_metrics, is_error, failed_samples


def process_file(
    output_file: str, temperature, temp_dir, num_samples, model, task_name, args
):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    document_ids_t, responses, responses_metrics, is_error, failed_samples = (
        process_temperature_file(
            temperature, temp_dir, num_samples, model, task_name, args
        )
    )

    if is_error:
        return

    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        # Define the column names
        column_names = (
            ["sample_id", "label_value", "label_name"]
            + [f"resp_{i}" for i in range(num_samples)]
            + [
                f"metric_{get_metric_name(task_name, args)}_{i}"
                for i in range(num_samples)
            ]
        )
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)  # , escapechar='\\')
        # Write the header
        writer.writerow(column_names)
        # Write the rows
        for doc_id in document_ids_t:
            responses_row = [
                responses[doc_id + "_" + str(id)] for id in range(num_samples)
            ]
            resps_metrics_row = [
                responses_metrics[doc_id + "_" + str(id)] for id in range(num_samples)
            ]

            row = [doc_id, temperature, "temp"] + responses_row + resps_metrics_row
            try:
                writer.writerow(row)
            except:
                pass
    print(f"CSV file '{output_file}' created successfully.")


def process_json(args) -> List[str]:

    tag = ""
    if args.metrics == "SentBert":
        tag = "_SentBert"

    # saves jsons for diversity scripts
    # returns the list of csv files to process
    parent_folder = args.samples_dir  # './samples'
    num_samples = args.num_samples  # 25
    num_docs = args.limit  # 1024
    model = args.model
    assert model is not None, "Model name must be provided"
    csv_files_to_generate = []

    tasks_to_process = []
    # with open(args.tmp_file, "w") as fmeta:
    for task_name in os.listdir(parent_folder):
        task_directory = os.path.join(parent_folder, task_name, str(num_docs))
        if not os.path.isdir(task_directory):
            logging.error(f"No samples found in {task_directory}")
            continue
        if not task_name == args.task_name:
            continue

        for sampling_method in os.listdir(task_directory):
            print(f"Processing {task_name} - {sampling_method}")
            sampling_directory = os.path.join(task_directory, sampling_method)
            if not os.path.isdir(sampling_directory):
                continue
            for sampling_method_2 in os.listdir(sampling_directory):
                print(f"Processing {sampling_method_2}")
                sampling_method_2_dir = os.path.join(
                    sampling_directory, sampling_method_2
                )
                ######## WILL SAVE HERE

                is_error = False

                for temparature_file in os.listdir(sampling_method_2_dir):

                    print(f"Processing temperature file: {temparature_file}")
                    if not temparature_file.startswith("temp_"):
                        continue

                    temperature = temparature_file[5:]
                    temp_dir = os.path.join(sampling_method_2_dir, temparature_file)
                    print("temp_dir", temp_dir)

                    # preserve the structure of "samples directory"
                    output_file = (
                        Path(sampling_method_2_dir)
                        / temparature_file
                        / "seed_0"
                        / model
                        / f"diversity_results{tag}"  # to separate SentBert from other metrics
                        / "data"
                        / "raw"
                        / get_file_name(args)
                    )
                    print(f"Output file: {output_file}")

                    # if exists, print and continue
                    tasks_to_process.append(str(output_file))

                    add_to_csv = True
                    if os.path.exists(output_file):
                        # print(f"CSV file '{output_file}' already exists.")
                        if not args.override:
                            add_to_csv = False
                        else:
                            print("Overriding existing file")

                    # append_args = [output_file, temperature, temp_dir, num_samples, model, task_name, args]
                    if add_to_csv:
                        print(f"Adding to csv: {output_file}")
                        # process the file
                        csv_files_to_generate.append(
                            (
                                output_file,
                                temperature,
                                temp_dir,
                                num_samples,
                                model,
                                task_name,
                                args,
                            )
                        )

                # Create the CSV file
                # output_file = f"diversity-eval/data/raw/output.{task_name}.{sampling_method}.{sampling_method_2}.csv"

                # fmeta.write(str(output_file))
                # fmeta.write("\n")

                # Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # process csv_files_to_generate in parallel
    n_cpus = min(os.cpu_count(), 16)
    with Pool(n_cpus) as p:
        p.starmap(process_file, csv_files_to_generate)

    return tasks_to_process


def get_script(mode, python_path, input_file, metrics):
    if "SentBert" in metrics:
        script = "run_diversity.sh"
    else:
        script = "run_diversity_cpu.sh"

    print("running", script)
    return f"{mode} {script} {str(PATH)}/diversity-eval/run_metrics.py --input_csv {input_file} --override --metrics {metrics}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only_correct_samples",
        action="store_true",
        help="Only include samples that are correct",
    )
    parser.add_argument(
        "--override", action="store_true", help="Override existing files"
    )
    parser.add_argument("--task_metric", type=str, default="exact_match")
    args = parser.parse_args()
    process_json(args)
