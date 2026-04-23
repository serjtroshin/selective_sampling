# Selective Sampling
This repository is the official implementation of ["Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs"](https://openreview.net/forum?id=IyOC5GCzv4).

# Installation Guide
This installation was tested using Red Hat Enterprise Linux (9.4) and CUDA Version 12.9.
- 1. Create a conda virtual environment:
```bash
conda create python=3.10.15 --name selective_sampling
conda activate selective_sampling
```
- 2. Install the required packages using Poetry:
```bash
poetry install
```
- 3. Patch the `vllm` to support selective sampling. This patch will copy selective_sampling/myllama.py to vllm models folder. While this is not neccessary (one can use the implemebtation based solely on logit processors), but more native implementation is neccessary to preserve efficiency.
```bash
python patch_vllm.py
```
- 4. Install the evaluation harness. This is lm-eval repo, with added custom model wrapper for vLLM.
```bash
cd lm-evaluation-harness
pip install -e .
```

# Usage (Symbolic GSM Example)
To evaluate the selective sampling, you can run the scripts from `run/` folder.
For example, to run the selective sampling with min_p sampling strategy, you can run:
```bash
bash run/run_simple_min_p.sh
```
To run the selective sampling with a classifier, you can run:
```bash
bash run/run_our_model.sh
```
To evaluate the diversity of the samples, you can run the following commands. Make sure to have 3 samples with seed from 0 to 2 in your output folder. Use 25 samples (seeds) for the full run, as well as limit >= number of task samples:
```bash
# for min_p sampling:
PYTHONPATH="." python run/run_diversity.py --task_name symbolic_gsm8k_cot_llama_main --samples_dir ./outputs/exps_paper_min_p --limit <limit> --model meta-llama__Meta-Llama-3.1-8B-Instruct --num_samples <number_of_samples>
# for our model e.g.:
PYTHONPATH="." python run/run_diversity.py --task_name symbolic_gsm8k_cot_llama_main --samples_dir ./outputs/exps_paper_ours --limit <limit> --model_from_seed_dir ./outputs/exps_paper_ours/symbolic_gsm8k_cot_llama_main/10/min_p/min_p_0.1/temp_1.0/seed_0 --num_samples <number_of_samples>
```

# Acknowledgements
This work was partly funded by the European Union's Horizon Europe (HE) Research and Innovation programme under Grant Agreement No 101070631 and from the UK Research and Innovation (UKRI) under the UK government's HE funding grant No 10039436.

This research was funded in part by the Netherlands Organization for Scientific Research (NWO) under project numbers VI.C.192.080 and 2023.017.

If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{selective_sampling2025,
  title={Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs},
  author={Troshin, Sergey and Mohammed, Wafaa and Meng, Yan and Monz, Christof and Fokkens, Antske and Niculae, Vlad},
  booktitle={COLM},
  year={2025},
  url={https://openreview.net/forum?id=IyOC5GCzv4}
}
