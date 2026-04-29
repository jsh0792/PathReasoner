train_file='/path/datasets/diagnosis_CoT.json'
test_file='/path/datasets/Diagnosis_CoT_test_format.json'
engine='nl'
model_name_or_path="/path/download/Qwen2.5-7B-Instruct"
ref_model_name_or_path="/path/download/Qwen2.5-7B-Instruct"
checkpoint="/path/src/results_sft/lr2e-5_ep3/pytorch_model.bin"

n_epochs='10'
kl_coef='0.05'
python_path="../main_GRPO_trl.py"

config_file="../default_config_deepspeed.yaml"
mkdir -p "${model_dir}"
mkdir -p "${log_dir}"

batch_size="1"
mini_batch_size="1"
eval_batch_size="2"
ppo_epochs="2"
num_workers="1"
learning_rate="3e-7"
weight_decay="0"
warmup_step="0"
clip_grad_norm="1"
vf_coef="5"
gamma="1.0"
lam="0.95"
adv_whitening='global'
seed="42"
max_input_length="700"
max_gen_length="700"
use_small_vocab='0'
keep_num_ckpt='0'

evaluating_epoch_freq="1"
logging_epoch_freq="1"
saving_epoch_freq="1"

logging_step_freq="1"
evaluating_step_freq="-100"
saving_step_freq="-100"

wandb_log="True"
wandb_project="R3_llama"
wandb_run_name="${exp_name}"

num_processes='2'
main_process_port='8888'
echo ${model_dir}
echo $python_path

export CUDA_VISIBLE_DEVICES=1,2,3,4
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True

accelerate launch \
	--config_file "${config_file}" \
	--num_processes=${num_processes} \
	--main_process_port=${main_process_port} \
$python_path \
	--model_name_or_path "${model_name_or_path}" \
	--tokenizer_name_or_path "${tokenizer_name_or_path}" \
	--ref_model_name_or_path "${ref_model_name_or_path}" \
	--checkpoint "${checkpoint}" \
	--train_file "${train_file}" \
	--test_file "${test_file}" \
	--model_dir "${model_dir}" \
	--batch_size "${batch_size}" \
	--mini_batch_size "${mini_batch_size}" \
	--ppo_epochs "${ppo_epochs}" \
	--n_epochs "${n_epochs}" \
	--num_workers "${num_workers}" \
	--learning_rate "${learning_rate}" \
	--weight_decay "${weight_decay}" \
	--warmup_step "${warmup_step}" \
	--clip_grad_norm "${clip_grad_norm}" \
	--vf_coef "${vf_coef}" \
	--kl_coef "${kl_coef}" \
	--gamma "${gamma}" \
	--lam "${lam}" \
	--evaluating_epoch_freq "${evaluating_epoch_freq}" \
	--logging_epoch_freq "${logging_epoch_freq}" \
	--saving_epoch_freq "${saving_epoch_freq}" \
	--evaluating_step_freq "${evaluating_step_freq}" \
	--logging_step_freq "${logging_step_freq}" \
	--saving_step_freq "${saving_step_freq}" \
	--seed "${seed}" \
	--max_input_length "${max_input_length}" \
	--max_gen_length "${max_gen_length}" \
	--wandb_log "${wandb_log}" \
	--wandb_project "${wandb_project}" \
	--wandb_run_name "${wandb_run_name}" \
	--engine "${engine}" \
	--use_small_vocab "${use_small_vocab}" \
	--adv_whitening "${adv_whitening}" \
	--keep_num_ckpt "${keep_num_ckpt}" \
	> "${log_dir}"/"${exp_name}".log 2>&1