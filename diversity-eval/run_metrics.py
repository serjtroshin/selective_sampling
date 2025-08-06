import argparse
from collections import defaultdict
import os
import csv
import sys
import inspect
from copy import deepcopy

from tqdm import tqdm
import json

# locals
import utils
import diversity_metrics


def decode_resp(response):
    response = response.replace("\\n", "\n")
    return response


def calc_metrics(params):

    # choose metrics and configurations
    metrics_dict = {}
    if params.metrics == "":
        # calc all metrics
        for name, obj in inspect.getmembers(sys.modules["diversity_metrics"]):
            if inspect.isclass(obj) and obj.use_me:
                metrics_dict.update({obj: {"name": name}})
    else:
        metrics_split = params.metrics.split(",")
        assert len(metrics_split) > 0, "No metrics to calculate."
        for metric in metrics_split:
            metrics_dict.update(
                {getattr(sys.modules["diversity_metrics"], metric): {"name": metric}}
            )

    for metric, metric_params in metrics_dict.items():
        metric_params.update(
            {
                "field_name": "metric_"
                + utils.CamleCase2snake_case(metric_params["name"]),
                "config": deepcopy(metric.default_config),
            }
        )
        if params.ignore_cache:
            metric_params["config"].update({"ignore_cache": True})
        # TODO - if one in the future will want to inject non-default metric config, here is the place

    print("#" * 30)
    print("Metrics parsing and validation done. Will calc the following metrics:")
    utils.dict_print(metrics_dict)

    # parse input csv path
    # will simply add .with_metrics.csv
    csv_dict = {
        path: {}
        for path in utils.parse_path_list(
            params.input_csv, default_path=utils.RAW_DATA_DIR, file_extension=".csv"
        )
    }

    # validating files
    for path, param_dict in csv_dict.items():
        with open(path, "r+", encoding="utf-8") as input_csv_f:
            reader = csv.DictReader(input_csv_f)
            assert (
                "sample_id" in reader.fieldnames
            ), "[{}] missing sample_id field".format(path)

            # validate resp fields
            resp_fields = [
                e
                for e in reader.fieldnames
                if e.startswith("resp_")
                and utils.represents_int(e.replace("resp_", ""))
            ]
            resp_ints = sorted([int(e.replace("resp_", "")) for e in resp_fields])
            assert len(resp_fields) > 0, "[{}] missing resp_i fields".format(path)
            assert resp_ints == list(
                range(len(resp_ints))
            ), "[{}] missing indices in resp_i fields".format(path)
            param_dict.update({"samples_per_set": len(resp_fields)})

            # validate metric fields
            metric_fields = [e for e in reader.fieldnames if e.startswith("metric_")]
            assert len(metric_fields) > 0, "[{}] missing metric_i fields".format(path)
            assert len(metric_fields) == len(
                resp_fields
            ), "[{}] missing indices in metric_i fields".format(path)

            # configure num_sets
            num_sets = 0
            for _ in reader:
                num_sets += 1
            assert num_sets > 0, "[{}] no samples in file".format(path)
            param_dict.update({"num_sets": num_sets})

            # configure out_path
            if utils.RAW_DATA_DIR in path:
                param_dict.update(
                    {
                        "out_path": path.replace(
                            utils.RAW_DATA_DIR, utils.METRICS_DATA_DIR
                        )
                    }
                )
            else:
                param_dict.update(
                    {
                        "out_path": os.path.join(
                            utils.METRICS_DATA_DIR, os.path.basename(path)
                        )
                    }
                )
            param_dict.update(
                {"run": (not os.path.isfile(param_dict["out_path"]) or params.override)}
            )

            # configure local metrics
            local_metrics = deepcopy(
                {
                    metric: metric_params
                    for metric, metric_params in metrics_dict.items()
                    if metric_params["field_name"] not in reader.fieldnames
                }
            )
            for metric, metric_params in local_metrics.items():
                if "input_path" in metric_params["config"].keys():
                    metric_params["config"]["input_path"] = path
                if "num_sets" in metric_params["config"].keys():
                    metric_params["config"]["num_sets"] = param_dict["num_sets"]
                if "samples_per_set" in metric_params["config"].keys():
                    metric_params["config"]["samples_per_set"] = param_dict[
                        "samples_per_set"
                    ]
                    ####### Edit configurations for the metric

                metric_params.update({"instance": metric(metric_params["config"])})
            param_dict.update({"metrics_to_calc": local_metrics})

    print("#" * 30)
    print("Parsing and validation done. Will calc metrics for the following files:")
    for k, v in csv_dict.items():
        if v["run"]:
            print(k + " -> " + v["out_path"])

    # calc metrics for each file
    for path, param_dict in csv_dict.items():
        if param_dict["run"]:
            # handle files
            out_dir = os.path.dirname(param_dict["out_path"])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # handle reader and writer
            input_csv_f = open(path, "r+", encoding="utf-8")
            # TODO - check if outfile exists
            output_csv_f = open(param_dict["out_path"], "w", encoding="utf-8")
            reader = csv.DictReader(input_csv_f)
            out_fields = reader.fieldnames + [
                metric_params["field_name"]
                for metric_params in param_dict["metrics_to_calc"].values()
            ]
            writer = csv.DictWriter(output_csv_f, out_fields)
            writer.writeheader()
            # if "gsm8k" in path or "minerva_math_intermediate_algebra" in path:
            #     metric_name = "exact_match"
            # elif "xsum" in path:
            #     metric_name = "bertscore_f1"
            # else:
            #     raise ValueError(f"Task {task_name} not supported")

            # write rows
            for idx, in_row in enumerate(reader):
                resp_set = [
                    decode_resp(in_row["resp_{}".format(i)])
                    for i in range(param_dict["samples_per_set"])
                ]
                # metric_set = [float(in_row['metric_{}_{}'.format(metric_name, i)]) for i in range(param_dict['samples_per_set'])]
                out_row = deepcopy(in_row)
                for metric, metric_params in param_dict["metrics_to_calc"].items():
                    metric_input = (
                        idx if metric.required_input == "set_index" else (resp_set)
                    )
                    output = metric_params["instance"](metric_input)
                    if output is None:

                        out_row.update({metric_params["field_name"]: "None"})
                    else:
                        out_row.update(
                            {metric_params["field_name"]: "{:.3f}".format(output)}
                        )
                writer.writerow(out_row)

            input_csv_f.close()
            output_csv_f.close()

            # print the average diversity metrics per label value
            label_value_2_metrics = {}
            print("Average diversity metrics for file: {}".format(path))
            write_result_to = param_dict["out_path"].replace(".csv", "_average.json")
            with (
                open(param_dict["out_path"], "r+", encoding="utf-8") as output_csv_f,
                open(write_result_to, "w") as write_f,
            ):
                result_dict = {}
                reader = csv.DictReader(output_csv_f)
                metrics_sum = {
                    metric_params["field_name"]: defaultdict(list)
                    for metric_params in param_dict["metrics_to_calc"].values()
                }
                # metric -> label value -> metric values
                for idx, in_row in enumerate(reader):
                    label_value = in_row["label_value"]
                    for metric_params in param_dict["metrics_to_calc"].values():
                        metrics_sum[metric_params["field_name"]][label_value].append(
                            float(in_row[metric_params["field_name"]])
                        )
                for metric_params in param_dict["metrics_to_calc"].values():

                    # print for each label value
                    print("---------- diversity metric -----------")
                    print(metric_params["field_name"])
                    for label_value, metric_values in metrics_sum[
                        metric_params["field_name"]
                    ].items():
                        print(
                            "temperature (label value): {} -> mean: {:.3f}".format(
                                label_value, sum(metric_values) / len(metric_values)
                            )
                        )
                # create result_dict
                for metric_params in param_dict["metrics_to_calc"].values():
                    result_dict[metric_params["field_name"]] = {
                        label_value: sum(metric_values) / len(metric_values)
                        for label_value, metric_values in metrics_sum[
                            metric_params["field_name"]
                        ].items()
                    }

                write_f.write(json.dumps(result_dict, indent=4))
            print("Average diversity metrics written to: {}".format(write_result_to))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate metrics for input csv files and output"
        " result to another csv file."
    )

    # main parameters
    parser.add_argument(
        "--input_csv",
        type=str,
        default="",
        help="Input results csv file. Support multiple, comma separated, or dirs. "
        "By default, will handle all files in ./data/raw/",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Metrics to calculate (by their class name). Support multiple, comma separated. "
        "By default, will use all available metrics from diversity_metrics.py",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="If true, will ignore existing cache file.",
    )
    parser.add_argument(
        "--override", action="store_true", help="If true, will override existing files."
    )

    params = parser.parse_args()
    utils.download_and_place_data()
    calc_metrics(params)
