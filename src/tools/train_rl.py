from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
from src.python_engine import run_python_code, process_code
from src.python_stdout_engine import run_python_stdout_code, compare_both_string_and_number_format, number_it
from src.utils import set_seed, is_numeric, timeout, discount_cumsum, do_gather
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import numpy as np
import wandb
import shutil
from prettytable import PrettyTable
from peft import LoraConfig, get_peft_model
import torch
from transformers import HfArgumentParser
from peft import LoraConfig
tqdm = partial(tqdm, ncols=0, leave=False)
TIMEOUT = 2


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    return variance

def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whiten = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whiten += mean
    return whiten

def logprobs_from_logits(logits, labels):
    if labels.device != logits.device:
        labels = labels.to(logits.device)

    logp = torch.nn.functional.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    
    return logpy.to("cuda")

def reward_func(completions, **kwargs):
    rewards = []
    for c in completions:
        if "correct_answer" in c:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def prepare_deepspeed_ref_model(model):
    # Adopted from: https://github.com/huggingface/trl/blob/02f5c1d8cee73045c837d01d7f1577a57779b035/trl/trainer/ppo_trainer.py#L1399
    import deepspeed

    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def prepare_cot_info(src_name):
    assert src_name in ['gsm8k', 'svamp']

    # default for common datasets
    instruction = 'Question:\n'
    cot_trigger = '\nAnswer reasoning:\n'
    answer_trigger = '\nTherefore, the answer is: '


    post_process_final_answer_fn_mapper = {
        'gsm8k': lambda x: float(x.replace(',', '').strip()),
        'svamp': lambda x: float(x.replace(',', '').strip()),
    }
    post_process_completed_question_answer_fn_mapper = {
        ('python', 'gsm8k'): lambda completed_question_answer: float(run_python_code(code_gen=completed_question_answer.split(cot_trigger)[-1].strip())),
        ('python', 'svamp'): lambda completed_question_answer: float(run_python_code(code_gen=completed_question_answer.split(cot_trigger)[-1].strip())),
        
        ('nl', 'gsm8k'): lambda completed_question_answer: float(completed_question_answer.split(cot_trigger)[-1].split(answer_trigger)[-1].strip()),
        ('nl', 'svamp'): lambda completed_question_answer: float(completed_question_answer.split(cot_trigger)[-1].split(answer_trigger)[-1].strip()),
    }
    compare_answer_fn_mapper = {
        'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
        'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    }

    return {
        'instruction': instruction,
        'cot_trigger': cot_trigger,
        'answer_trigger': answer_trigger,
        'post_process_final_answer_fn_mapper': post_process_final_answer_fn_mapper,
        'post_process_completed_question_answer_fn_mapper': post_process_completed_question_answer_fn_mapper,
        'compare_answer_fn_mapper': compare_answer_fn_mapper,
    }

def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        raw_dataset = DatasetDict({
            'train': Dataset.from_list(json.load(open(args['train_file'], 'r')))
                        if not args['train_file'].rstrip('/').endswith('_cache')
                        else load_from_disk(args['train_file']),
            'test':  Dataset.from_list(json.load(open(args['test_file'], 'r')))
                        if not args['test_file'].rstrip('/').endswith('_cache')
                        else load_from_disk(args['test_file']),
        })
        accelerator.print('Raw data:', raw_dataset)

        src_name = raw_dataset['train']['item_id'][0].split('_')[0]
        cot_info = prepare_cot_info(src_name)
        instruction   = cot_info['instruction']
        cot_trigger   = cot_info['cot_trigger']
        answer_trigger = cot_info['answer_trigger']

        def tokenize_fn(batch, args, tokenizer):
            assert tokenizer.eos_token_id is not None
            new_batch = defaultdict(list)
            all_keys = list(batch.keys())
            for item_values in zip(*(batch[k] for k in all_keys)):
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                item_id     = item['item_id']
                question    = item['question']
                answer_value = item['answer_value']
                answer_cot  = item.get('answer_cot', None)

                # === NEW === read optional fields for entity/semantic rewards
                gt_entities    = item.get('gt_entities', [])
                gt_answer_text = item.get('gt_answer_text', None) or (answer_cot or '')

                if answer_value is not None:
                    answer_value = answer_value.strip()

                if answer_cot:
                    if args['engine'] == 'nl' and src_name in ['gsm8k']:
                        answer_cot += f'{answer_trigger} {answer_value}'
                    input       = f'{instruction}{question}'
                    output      = f'{answer_cot}'
                    prefix_text = f'{instruction}{question}'
                    if answer_cot.startswith('def'):
                        question_1 = question.replace("\n\n### Response:", "")
                        if src_name in ['gsm8k', 'svamp'] and args['engine'] == 'python':
                            prefix_text += f'def solution():\n    """{question_1}"""\n'
                else:
                    input       = f'{instruction}{question}{cot_trigger}'
                    output      = f'{answer_cot}'
                    prefix_text = f'{instruction}{question}{cot_trigger}'
                    if src_name in ['gsm8k', 'svamp'] and args['engine'] == 'python':
                        prefix_text += f'def solution():\n    """{question}"""\n'

                input_encode  = tokenizer(input, add_special_tokens=False)
                output_encode = tokenizer(output, add_special_tokens=False)
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

                input_ids    = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                labels       = [-100] * len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                prefix       = prefix_encode['input_ids']
                prefix_attention_mask = prefix_encode['attention_mask']

                input_ids    = input_ids[:args['max_input_length']]
                labels       = labels[:args['max_input_length']]
                attention_mask = attention_mask[:args['max_input_length']]
                prefix       = prefix[:args['max_input_length']]
                prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]

                new_batch['input_ids'].append(input_ids)
                new_batch['labels'].append(labels)
                new_batch['attention_mask'].append(attention_mask)
                new_batch['prefix'].append(prefix)
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)
                new_batch['item_id'].append(item_id)
                new_batch['question'].append(question)
                new_batch['prefix_text'].append(prefix_text)
                new_batch['answer_cot'].append(answer_cot)
                new_batch['answer_value'].append(answer_value)
                # === NEW ===
                new_batch['gt_entities'].append(gt_entities)
                new_batch['gt_answer_text'].append(gt_answer_text)

            return new_batch

        tokenized_dataset = DatasetDict({
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True,
                remove_columns=dataset.column_names,
                num_proc=None, load_from_cache_file=True, keep_in_memory=False,
            ) for mode, dataset in raw_dataset.items()
        })
        accelerator.print('Processed data:', tokenized_dataset)

        if accelerator.is_main_process and args['wandb_log']:
            wandb.config.update({
                "src_name": src_name,
                "instruction": instruction,
                "cot_trigger": cot_trigger,
                "answer_trigger": answer_trigger,
                "raw_dataset": str(raw_dataset),
                "tokenized_dataset": str(tokenized_dataset),
            })

    train_dataloader = DataLoader(
        tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'],
        num_workers=args['num_workers'], pin_memory=True,
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer),
    )
    test_dataloader = DataLoader(
        tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'],
        num_workers=args['num_workers'], pin_memory=True,
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer),
    )

    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader), cot_info

def collate_fn(batch, args, tokenizer):
    max_input_length  = max(len(item['input_ids']) for item in batch)
    max_target_length = max(len(item['labels'])    for item in batch)
    max_prefix_length = max(len(item['prefix'])    for item in batch)

    labels_left_padded, prefix_left_padded, prefix_attention_mask_left_padded = [], [], []
    for item in batch:
        labels_left_padded.append([-100] * (max_target_length - len(item['labels'])) + item['labels'])
        prefix_left_padded.append([tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix'])
        prefix_attention_mask_left_padded.append(
            [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask']
        )

    ppo_forward_kwargs = {
        'query': [item['prefix_text'] for item in batch],
        'query_tensors': torch.LongTensor(prefix_left_padded),
        'query_tensors_attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'answer_values': [item['answer_value'].replace(',', '') for item in batch],
        'item_ids': torch.LongTensor([int(item['item_id'].split('_')[1]) for item in batch]),
        # === NEW ===
        'gt_entities': [list(item['gt_entities']) for item in batch],
        'gt_answers':  [item['gt_answer_text']    for item in batch],
    }
    generate_prefix_kwargs = {
        'input_ids': torch.LongTensor(prefix_left_padded),
        'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
        'labels': torch.LongTensor(labels_left_padded),
    }
    return {'ppo_forward_kwargs': ppo_forward_kwargs, 'generate_prefix_kwargs': generate_prefix_kwargs}

def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    tokenizer.save_pretrained(save_path)
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            # os.remove(ckpt_to_be_removed)
            shutil.rmtree(ckpt_to_be_removed)


def allgather(tensor, group=None):
    """smantic sugar for torch.distributed.all_gather.

    Args:
        tensor: (bs, ...)
        group:

    Returns:
        All gathered tensor (world_size, bs, ...)
    """
    if group is None:
        group = torch.distributed.group.WORLD
    allgather_tensor = [torch.zeros_like(tensor) for _ in range(group.size())]
    torch.distributed.all_gather(allgather_tensor, tensor, group=group)
    allgather_tensor = torch.stack(allgather_tensor, dim=0)
    return allgather_tensor


def allgather_masked_whiten(values, mask, shift_mean=False):
    """Whiten values with all-gathered masked values.

    Args:
        values: (bs, ...)
        mask: (bs, ...)
        shift_mean: bool

    Returns:
        whitened values, (bs, ...)
    """
    allgather_values = allgather(values)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_values {allgather_values.shape}, {allgather_values[0, 0:3]}')

    allgather_mask = allgather(mask)  # (n_proc, bs, ...)
    # accelerator.print(f'allgather_mask {allgather_mask.shape}, {allgather_mask[0, 0:3]}')

    global_mean = masked_mean(allgather_values, allgather_mask)
    global_var = masked_var(allgather_values, allgather_mask)
    whitened = (values - global_mean) * torch.rsqrt(global_var + 1e-8)
    if shift_mean:
        whitened += global_mean
    return whitened


def logging_values(ids, vpreds, rets, advs, old_vpreds, rews, score_rews, masks, tokenizer):
    

    get_str_digits = lambda x: str(round(x.item(), 3))

    for tmp_i in range(ids.size(0)):
        mk_ids = torch.masked_select(ids[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_vpreds = torch.masked_select(vpreds[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_rets = torch.masked_select(rets[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_advs = torch.masked_select(advs[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_old_vpreds = torch.masked_select(old_vpreds[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_rews = torch.masked_select(rews[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)
        mk_score_rews = torch.masked_select(score_rews[tmp_i], masks[tmp_i].bool())  # (valid_resp_len,)

        accelerator.print(tokenizer.decode(mk_ids))

        table = PrettyTable()
        table.field_names = ["a_t", "V(s_t)", "Ret(s_t)", "A(s_t,a_t)", "old_V(s_t)", "r(s_t,a_t)", "score_r(s_t,a_t)"]
        for tmp_j in range(mk_ids.nelement()):
            a_t = tokenizer.decode(mk_ids[tmp_j])
            table.add_row([
                a_t if a_t != '\n' else '\\n', 
                get_str_digits(mk_vpreds[tmp_j]), 
                get_str_digits(mk_rets[tmp_j]), 
                get_str_digits(mk_advs[tmp_j]), 
                get_str_digits(mk_old_vpreds[tmp_j]),
                get_str_digits(mk_rews[tmp_j]),
                get_str_digits(mk_score_rews[tmp_j])
            ])
        table_str = table.get_string()
        accelerator.print(table_str)
        accelerator.print('\n')


def rollout(args, model, ref_model, tokenizer,
            query_tensors, query_tensors_attention_mask,
            answer_values, src_name, cot_info,
            gt_entities_list, gt_answers_list,
            entity_matcher, semantic_scorer, entity_sim_fn=None):

    model.eval()
    with torch.no_grad():
        gen_output = accelerator.unwrap_model(model).generate(
            input_ids=query_tensors,
            attention_mask=query_tensors_attention_mask,
            top_k=0.0, top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args['max_gen_length'],
        )
        completed_tensors = gen_output
        completed_tensors = pad_across_processes(
            completed_tensors, dim=1,
            pad_index=tokenizer.pad_token_id, pad_first=False,
        )

    decoded_texts = [
        tokenizer.decode(ids.cpu().numpy().tolist(), skip_special_tokens=True)
        for ids in completed_tensors
    ]
    correctness, reward_components = compute_multi_granular_rewards(
        decoded_texts=decoded_texts,
        gt_entities_list=gt_entities_list,
        gt_answers_list=gt_answers_list,
        entity_matcher=entity_matcher,
        semantic_scorer=semantic_scorer,
        alpha=args.get('entity_reward_coef', 0.5),
        beta=args.get('entity_soft_beta', 0.5),
        entity_sim_fn=entity_sim_fn,
    )

    model_input_ids = completed_tensors
    model_attention_mask = (completed_tensors != tokenizer.pad_token_id)
    with torch.no_grad():
        outputs = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        lm_logits = outputs.logits
        old_logprob = logprobs_from_logits(lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])

        ref_logprob = None
        if ref_model is not None:
            ref_outputs = ref_model(input_ids=model_input_ids, attention_mask=model_attention_mask)
            ref_lm_logits = ref_outputs[0] if isinstance(ref_outputs, tuple) else ref_outputs.logits
            ref_logprob = logprobs_from_logits(ref_lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])

    prompt_len = query_tensors.size(1)
    seq_len    = model_input_ids.size(1)

    mask = torch.zeros((model_input_ids.size(0), seq_len),
                       dtype=torch.bool, device=model_input_ids.device)
    score_rew = torch.zeros((model_input_ids.size(0), seq_len),
                            dtype=torch.float, device=model_input_ids.device)
    mask[:, prompt_len - 1:] = 1
    if tokenizer.pad_token_id is not None:
        mask[model_input_ids == tokenizer.pad_token_id] = 0

    eos_mask = (model_input_ids == tokenizer.eos_token_id)
    first_eos_indices = torch.argmax(eos_mask.float(), dim=1)
    has_eos = eos_mask.any(dim=1)
    end_indices = torch.where(
        has_eos, first_eos_indices,
        torch.tensor(seq_len - 1, device=model_input_ids.device),
    )

    for i in range(model_input_ids.size(0)):
        end_idx = end_indices[i].item()
        if end_idx + 1 < seq_len:
            mask[i, end_idx + 1:] = 0
        reward_pos = min(end_idx, seq_len - 1)
        reward_pos = max(reward_pos, prompt_len - 1)
        mask[i, reward_pos] = 1
        score_rew[i, reward_pos] = correctness[i]

    kl_rew = None
    rew = score_rew.clone()
    if ref_logprob is not None:
        kl_div = old_logprob - ref_logprob
        mask_shifted = mask[:, 1:]
        kl_div_masked = kl_div * mask_shifted
        kl_rew_tensor = torch.zeros_like(score_rew)
        kl_rew_tensor[:, 1:] = -kl_div_masked
        kl_coef = args["kl_coef"]
        rew = rew + kl_coef * kl_rew_tensor
        kl_rew = kl_rew_tensor

    rew = rew.to(dtype=old_logprob.dtype) * mask
    score_rew = score_rew.to(dtype=old_logprob.dtype) * mask
    if kl_rew is not None:
        kl_rew = kl_rew.to(dtype=old_logprob.dtype) * mask
    old_logprob = old_logprob * mask[:, 1:]

    model.train()
    return (model_input_ids, model_attention_mask, mask,
            rew, score_rew, kl_rew, correctness,
            old_logprob, ref_logprob,
            reward_components)

def train_one_epoch(args, model, ref_model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, global_iter_num, test_dataset, test_dataloader, cot_info,
                    prefix, epoch, best_eval_log_dict, most_recent_ckpts_paths,
                    entity_matcher, semantic_scorer, entity_sim_fn=None):  # === NEW args ===
    max_epoch = args['n_epochs']
    model_dir = args['model_dir']
    clip_grad_norm = args.get('clip_grad_norm', None)
    vf_coef = args['vf_coef']
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    logging_step_freq    = args.get('logging_step_freq', None)
    saving_step_freq     = args.get('saving_step_freq', None)
    group_size           = args.get('group_size', 4)

    model.train()
    epoch_result_dict = defaultdict(list)
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                           disable=not accelerator.is_main_process, desc='Train Loop'):
        result_dict = defaultdict(list)
        q_tensors   = batch['ppo_forward_kwargs']['query_tensors']
        q_mask      = batch['ppo_forward_kwargs']['query_tensors_attention_mask']
        ans_values  = batch['ppo_forward_kwargs']['answer_values']
        gt_ents     = batch['ppo_forward_kwargs']['gt_entities']   # === NEW
        gt_answers  = batch['ppo_forward_kwargs']['gt_answers']    # === NEW

        batch['ppo_forward_kwargs']['query_tensors']                = q_tensors.repeat_interleave(group_size, dim=0)
        batch['ppo_forward_kwargs']['query_tensors_attention_mask'] = q_mask.repeat_interleave(group_size, dim=0)
        batch['ppo_forward_kwargs']['answer_values'] = list(itertools.chain.from_iterable(
            itertools.repeat(x, group_size) for x in ans_values))
        # === NEW ===
        batch['ppo_forward_kwargs']['gt_entities'] = list(itertools.chain.from_iterable(
            itertools.repeat(x, group_size) for x in gt_ents))
        batch['ppo_forward_kwargs']['gt_answers']  = list(itertools.chain.from_iterable(
            itertools.repeat(x, group_size) for x in gt_answers))

        model.eval()
        (model_input_ids, model_attention_mask, mask,
         rew, score_rew, kl_rew, correctness,
         old_logprob, ref_logprob,
         reward_components) = rollout(
            args, model, ref_model, tokenizer,
            query_tensors=batch['ppo_forward_kwargs']['query_tensors'],
            query_tensors_attention_mask=batch['ppo_forward_kwargs']['query_tensors_attention_mask'],
            answer_values=batch['ppo_forward_kwargs']['answer_values'],
            src_name=train_dataset[0]['item_id'].split('_')[0],
            cot_info=cot_info,
            gt_entities_list=batch['ppo_forward_kwargs']['gt_entities'],
            gt_answers_list=batch['ppo_forward_kwargs']['gt_answers'],
            entity_matcher=entity_matcher,
            semantic_scorer=semantic_scorer,
            entity_sim_fn=entity_sim_fn,
        )

        # === NEW === log per-component rewards
        if accelerator.is_main_process and args['wandb_log']:
            wandb.log({
                "reward/r_format":   float(np.mean(reward_components['r_format'])),
                "reward/r_semantic": float(np.mean(reward_components['r_semantic'])),
                "reward/r_entity":   float(np.mean(reward_components['r_entity'])),
                "reward/r_total":    float(np.mean(correctness)),
            }, step=global_iter_num)

        if accelerator.is_main_process:
            print(f"\n[DEBUG DISPLAY] --- Step {global_step} ---", flush=True)
            full_text = tokenizer.decode(model_input_ids[0], skip_special_tokens=False)
            print(">>> Full Sequence (Query + Response):")
            print(full_text)
            if correctness is not None:
                print(f">>> Correctness: {correctness[0]}  "
                      f"(fmt={reward_components['r_format'][0]}, "
                      f"sem={reward_components['r_semantic'][0]}, "
                      f"ent={reward_components['r_entity'][0]})")

        model.train()

        if rew.dim() > 1:
            step_rewards = rew.sum(dim=1)
        else:
            step_rewards = rew
        rewards_grouped = step_rewards.view(-1, group_size)
        mean_rew = rewards_grouped.mean(dim=1, keepdim=True)
        std_rew  = rewards_grouped.std(dim=1, keepdim=True)
        adv_scalar = (rewards_grouped - mean_rew) / (std_rew + 1e-8)
        adv_scalar = adv_scalar.view(-1)
        resp_len_per_sample = torch.clamp(mask.sum(dim=1), min=1.0)
        adv = adv_scalar.unsqueeze(1).expand_as(mask).clone() * mask

        batch_size_per_gpu      = len(batch['ppo_forward_kwargs']['query'])
        mini_batch_size_per_gpu = args["mini_batch_size"]
        ppo_epochs              = args["ppo_epochs"]

        train_stats = {}
        for _ in range(ppo_epochs):
            perms = torch.randperm(batch_size_per_gpu)
            for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                b_inds = perms[mini_idx: mini_idx + mini_batch_size_per_gpu]
                cur_model_input_ids = model_input_ids[b_inds].to(accelerator.device)
                cur_model_attention_mask = model_attention_mask[b_inds].to(accelerator.device)
                cur_old_logprob = old_logprob[b_inds].to(accelerator.device)
                cur_mask        = mask[b_inds].to(accelerator.device)
                cur_adv         = adv[b_inds].to(accelerator.device)
                cur_resp_len    = resp_len_per_sample[b_inds].to(accelerator.device)

                model.train()
                outputs   = model(input_ids=cur_model_input_ids, attention_mask=cur_model_attention_mask)
                lm_logits = outputs.logits[:, :-1, :]
                shift_labels = cur_model_input_ids[:, 1:]
                logprob  = logprobs_from_logits(lm_logits, shift_labels)

                target_device = lm_logits.device
                logprob         = logprob.to(target_device)
                cur_old_logprob = cur_old_logprob.to(target_device)
                cur_adv         = cur_adv.to(target_device)
                cur_mask        = cur_mask.to(target_device)

                if cur_old_logprob.shape[1] == cur_model_input_ids.shape[1]:
                    cur_old_logprob_aligned = cur_old_logprob[:, 1:]
                else:
                    cur_old_logprob_aligned = cur_old_logprob
                cur_adv_aligned  = cur_adv[:, 1:]
                cur_mask_aligned = cur_mask[:, 1:]

                ratio = torch.exp(logprob - cur_old_logprob_aligned)
                pg_losses  = -cur_adv_aligned * ratio
                pg_losses2 = -cur_adv_aligned * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                pg_loss_per_token = torch.max(pg_losses, pg_losses2)
                pg_loss_sum = (pg_loss_per_token * cur_mask_aligned).sum(dim=-1)
                pg_loss = (pg_loss_sum / (cur_resp_len + 1e-8)).mean()

                loss = pg_loss
                epoch_result_dict['loss'].append(loss.item())

                if accelerator.distributed_type == "DEEPSPEED":
                    accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                    accelerator.deepspeed_engine_wrapped.engine.step()
                else:
                    accelerator.backward(loss)
                    if clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

                n_correct, total = do_gather([sum(correctness), len(correctness)])
                train_stats["acc"]     = n_correct / total
                train_stats["ncor"]    = n_correct
                train_stats["total"]   = total
                train_stats['pg_loss'] = pg_loss.item()
                for k, v in train_stats.items():
                    result_dict[k].append(v)

                global_iter_num += 1

        scheduler.step()
        global_step += 1

        epoch_result_dict['loss'].append(loss.item())
        for k, v in train_stats.items():
            epoch_result_dict[k].append(v)

        eval_log_dict = {}
        is_best = False
        if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
            evaluate_result_dict = {
                f'Eval.Gen.{k}': v
                for k, v in evaluate_generation(args, model, test_dataset, test_dataloader,
                                                tokenizer, cot_info).items()
            }
            eval_log_dict.update(evaluate_result_dict)
            if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                is_best = True
                best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']

        train_log_dict = {}
        if logging_step_freq is not None and global_step % logging_step_freq == 0:
            train_log_dict = {
                f'Train.{k}': sum(v) / len(v) if isinstance(v, list) else v
                for k, v in epoch_result_dict.items()
            }

        if eval_log_dict or train_log_dict:
            log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
            if accelerator.is_main_process and args['wandb_log']:
                wandb.log(log_dict, step=global_iter_num)
                log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'] + '|' + wandb.run.id, **log_dict}
            log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
            accelerator.print(f"{prefix}[Epoch={epoch}/{max_epoch}, Step={global_step}] {log_dict}")

        if saving_step_freq is not None and global_step % saving_step_freq == 0:
            if is_best:
                do_checkpoint(args, model, tokenizer, os.path.join(model_dir, f'best'))
            if args['keep_num_ckpt'] > 0:
                do_checkpoint(args, model, tokenizer,
                              os.path.join(model_dir, f'global_step_{str(global_step)}'),
                              most_recent_ckpts_paths)

        for k, v in epoch_result_dict.items():
            if len(v) > 1:
                epoch_result_dict[k] = v[-1:]

    epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    return epoch_result_dict, global_step, global_iter_num


def load_custom_llm(args, device_map="cpu"):
    llm_name_or_path = args['model_name_or_path']
    checkpoint_path = args['checkpoint']
    use_llm_lora = True 

    print(f"Loading base model from {llm_name_or_path}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    if use_llm_lora:
        print("Applying LoRA config...", flush=True)
        peft_config = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.1, bias='none', task_type='CAUSAL_LM',
        )
        base_model = get_peft_model(base_model, peft_config)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('llm.'):
                new_key = k[4:] 
                new_state_dict[new_key] = v
            elif 'visual_encoder' in k or 'projector' in k:
                continue
            else:
                new_state_dict[k] = v
        print("Custom weights loaded.")
        
    return base_model

def main(args):
    set_seed(args['seed'] + accelerator.process_index)

    if accelerator.is_main_process and args['wandb_log']:
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        wandb.config.update(args)

    # ---- tokenizer ----
    if args.get('use_small_vocab', False) and args.get('engine') == 'game24':
        from src.game24_tokenizer import Calc24Vocab32Tokenizer
        tokenizer = Calc24Vocab32Tokenizer()
    else:
        default_path = "/huggingface_download/Qwen2.5-7B-Instruct"
        model_path = args.get('tokenizer_name_or_path', default_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True, encode_special_tokens=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    (train_dataset, train_dataloader), (test_dataset, test_dataloader), cot_info = \
        prepare_datasets_and_data_loaders(args, tokenizer)

    entity_lexicon  = load_entity_lexicon(args['entity_lexicon_path'])
    entity_matcher  = build_entity_matcher(entity_lexicon)
    semantic_scorer = build_semantic_judger(
        endpoint_url=args['judge_endpoint_url'],
        model_name=args.get('judge_model_name', 'qwen2.5-32b-instruct'),
        concurrency=args.get('judge_concurrency', 8),
    )
    entity_sim_fn = None 
    
    train_base_model = load_custom_llm(args, device_map="cpu")
    model = train_base_model
    model.train()
    ref_base_model = load_custom_llm(args, device_map="cpu")
    ref_model = ref_base_model
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    n_epochs = args['n_epochs']
    batch_size_per_device = len(train_dataloader) // accelerator.num_processes
    num_training_steps = batch_size_per_device * n_epochs
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 \
        else int(0.1 * num_training_steps)

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad],
         "weight_decay": args['weight_decay']},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"]) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)

    accelerator.print(
        f"***** Running training *****\n"
        f"  Num examples = {len(train_dataset)}\n"
        f"  Num Epochs = {n_epochs}\n"
        f"  Instantaneous batch size per device = {args['batch_size']}\n"
        f"  Total train batch size = {args['batch_size'] * accelerator.num_processes}\n"
        f"  Total optimization steps = {num_training_steps}\n"
        f"  Warm up step: {warmup_step}\n"
        f"  Learning rate: {args['learning_rate']}\n"
    )

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader,
    )
    if ref_model is not None:
        if accelerator.distributed_type == "DEEPSPEED":
            ref_model = prepare_deepspeed_ref_model(ref_model)
        else:
            ref_model = accelerator.prepare(ref_model)

    global_step = 0
    global_iter_num = 0
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    logging_epoch_freq    = args['logging_epoch_freq']
    saving_epoch_freq     = args['saving_epoch_freq']
    model_dir = args['model_dir']
    best_eval_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    most_recent_ckpts_paths = []

    for epoch in range(1, n_epochs + 1):
        kwargs = {
            'args': args,
            'model': model,
            'ref_model': ref_model,
            'train_dataset': train_dataset,
            'train_dataloader': train_dataloader,
            'test_dataset': test_dataset,
            'test_dataloader': test_dataloader,
            'cot_info': cot_info,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'global_step': global_step,
            'global_iter_num': global_iter_num,
            'tokenizer': tokenizer,
            'prefix': '[Train-Step]',
            'epoch': epoch,
            'best_eval_log_dict': best_eval_log_dict,
            'most_recent_ckpts_paths': most_recent_ckpts_paths,
            # === NEW ===
            'entity_matcher':  entity_matcher,
            'semantic_scorer': semantic_scorer,
            'entity_sim_fn':   entity_sim_fn,
        }

        # === REMOVED === the stray GRPOTrainer(...).train() block that used to be here.
        # It ran a second training loop with a different config and caused state conflicts.

        train_epoch_result_dict, global_step, global_iter_num = train_one_epoch(**kwargs)

        eval_log_dict = {}
        is_best = False
        if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
            evaluate_result_dict = {
                f'Eval.Gen.{k}': v
                for k, v in evaluate_generation(args, model, test_dataset, test_dataloader,
                                                tokenizer, cot_info).items()
            }
            eval_log_dict.update(evaluate_result_dict)
            if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                is_best = True
                best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']

        train_log_dict = {}
        if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
            train_log_dict = {
                f'Train.{k}': sum(v) / len(v) if isinstance(v, list) else v
                for k, v in train_epoch_result_dict.items()
            }

        if eval_log_dict or train_log_dict:
            log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
            if accelerator.is_main_process and args['wandb_log']:
                wandb.log(log_dict, step=global_iter_num)
                log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'] + '|' + wandb.run.id, **log_dict}
            log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
            accelerator.print(f"[Epoch={epoch}/{args['n_epochs']}, Step={global_step}] {log_dict}")

        if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
            if is_best:
                do_checkpoint(args, model, tokenizer, os.path.join(model_dir, f'best'))
            if args['keep_num_ckpt'] > 0:
                do_checkpoint(args, model, tokenizer,
                              os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}'),
                              most_recent_ckpts_paths)

def evaluate_generation(args, model, dataset, dataloader, tokenizer, cot_info):
    model.eval()
    predictions = []
    targets = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process,
                           desc='Evaluation Gen Loop'):
        output_ = accelerator.unwrap_model(model).generate(
            **batch['generate_prefix_kwargs'],
            max_length=args['max_gen_length'],
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)

        labels = batch['generate_prefix_kwargs']['labels']
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        labels[labels == -100] = tokenizer.pad_token_id

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)

        preds = [tokenizer.decode(g.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in
                 generated_ids]
        predictions.extend(preds)
        target = [tokenizer.decode(t.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in
                  labels]
        targets.extend(target)

    predictions = predictions[:len(dataset)]
    targets = targets[:len(dataset)]

    post_process_final_answer_fn_mapper = cot_info['post_process_final_answer_fn_mapper']
    compare_answer_fn_mapper = cot_info['compare_answer_fn_mapper']
    post_process_completed_question_answer_fn_mapper = cot_info['post_process_completed_question_answer_fn_mapper']
    if accelerator.is_main_process and accelerator.is_local_main_process:
        results = [{
            'pred': pred,
            'tar': tar,
            'item_id': item.get('item_id', None),
            'answer_value': item.get('answer_value', None),
            'answer_type': item.get('answer_type', None),
        } for pred, tar, item in zip(predictions, targets, dataset)]

        corr_value = 0
        for cur_res in results:
            prediction, target, item_id = cur_res['pred'], cur_res['tar'], cur_res['item_id']
            src_name = item_id.split('_')[0]
            answer_value = cur_res['answer_value']

            ## Processing target
            target_cot = target.strip()
            target_value = post_process_final_answer_fn_mapper[src_name](answer_value)
            cur_res['target_cot'] = target_cot
            cur_res['target_value'] = target_value

            ## Processing prediction
            try:
                with timeout(seconds=TIMEOUT):
                    prediction_cot = prediction.strip()
                    prediction_value = post_process_completed_question_answer_fn_mapper[(args['engine'], src_name)](prediction_cot)
            except:
                prediction_cot = None
                prediction_value = None
            cur_res['prediction_cot'] = prediction_cot
            cur_res['prediction_value'] = prediction_value

            # Compute correctness
            is_correct = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
            corr_value += is_correct
            cur_res['is_correct'] = is_correct
        res_path = args['model_dir'].rstrip('/')+ '/' + '_res.json'
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=2)
        value_accuracy = corr_value / len(predictions) * 100
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]

    # Metric summary:
    model.train()
    return {'value_accuracy': value_accuracy}

if __name__ == '__main__':

    NONE_INT = -100
    NONE_STR = 'None'

    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str
        test_file: str
        checkpoint: str = field(default=NONE_STR)
        batch_size: int = field(default=8)
        mini_batch_size: int = field(default=8)
        eval_batch_size: int = field(default=8)
        ppo_epochs: int = field(default=1)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        vf_coef: float = field(default=1.0)
        kl_coef: float = field(default=0.1)
        gamma: float = field(default=0.98)
        lam: float = field(default=0.95)
        ref_model_name_or_path: str = field(default="")
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        logging_seq_str_step_freq: int = field(default=NONE_INT)
        logging_values_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        max_gen_length: int = field(default=700)
        keep_num_ckpt: int = field(default=5)
        separate_vf: int = field(default=0)
        init_value_model_with_rm: int = field(default=0)
        init_value_head_with_rm: int = field(default=0)
        rm_model_name_or_path: str = field(default="/your_model_path_here")
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default='tmp')
        wandb_run_name: str = field(default='default_run_name')
        engine: str = field(default='nl')
        use_small_vocab: int = field(default=0)
        adv_whitening: str = field(default='global')

        judge_model_name:    str = field(default='qwen2.5-32b-instruct')
        judge_concurrency:   int = field(default=8)
        entity_reward_coef:  float = field(default=0.5, metadata={"help": "alpha: weight on R_entity."})
        entity_soft_beta:    float = field(default=0.5, metadata={"help": "beta: weight on soft intersection term."})

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)