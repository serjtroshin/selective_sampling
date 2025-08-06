export DYNAMIC_TEMPERATURE_THRESHOLD=0.5  # threshold for the classifier
limit=10
# if using local path: pretrained local path: pretrained=/gpfs/work4/0/gus20642/gdg/paper/all_hiddens.gsm_symbolic_main_paper.fixed/v7.fixed_exact_match_cb_False/avg_risk__temp_3.0,dtype=bfloat16   
# might need to pre-cache model weight first by downloading serjtroshin/selective_sampling_gsm_symbolic HF repo
for seed in 0 1 2; do  # for full run, use 25 seeds and set limit to 5000
python -m lm_eval --model vllm_wrapper     --model_args pretrained=serjtroshin/selective_sampling_gsm_symbolic,dtype=bfloat16     --device cuda:0     --batch_size 256 --limit ${limit} --write_out --log_samples --output_path outputs/exps_paper_ours/symbolic_gsm8k_cot_llama_main/${limit}/min_p/min_p_0.1/temp_1.0/seed_${seed}/ --verbosity DEBUG --tasks symbolic_gsm8k_cot_llama_main --include_path selective_sampling/tasks --gen_kwargs sampling_config=min_p,override__min_p=0.1,override__temperature=1.0,seed=${seed},max_gen_toks=1024 --apply_chat_template --fewshot_as_multiturn
done