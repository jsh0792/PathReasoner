import torch
import numpy as np
import json
import os
import time
from peft import get_peft_model_state_dict

def make_serializable(obj):

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    else:
        return obj

def save_logs_to_jsonl(log_results, step_num, output_dir="./training_logs", rank=0, item_id=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"training_log_rank_{rank}.jsonl")
    
    with open(filename, "a", encoding="utf-8") as f:
        for log in log_results:
            log_entry = {}
            log_entry['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry['item_id'] = item_id
            log_entry['step'] = step_num
            log_entry['extracted_ans'] = log['extracted_ans']
            log_entry['target_value'] = log['target_value']
            log_entry['Qwen_score'] = log['Qwen_score']
            log_entry['final_score'] = log['final_score']
            log_entry['completed_text'] = log.get('completed_text')

            if isinstance(log_entry['extracted_ans'], str) and 'answer' in log_entry['extracted_ans']:
                log_entry['completed_text'] = log.get('completed_text')

            if log_entry['extracted_ans'] is None:
                log_entry['completed_text'] = log.get('completed_text')

            clean_entry = make_serializable(log_entry)
            
            try:
                f.write(json.dumps(clean_entry, ensure_ascii=False) + "\n")
            except TypeError as e:
                print(f"⚠️ Serialization Warning: Skipping a log line due to: {e}")


def save_weights_for_custom_loading(xtuner_model, longnet, projector, output_dir="./new_checkpoints"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    llm_lora_state_dict = get_peft_model_state_dict(xtuner_model.llm)
    
    new_lora_path = os.path.join(output_dir, 'llm_lora.pth')
    torch.save(llm_lora_state_dict, new_lora_path)
    longnet_path = os.path.join(output_dir, 'longnet_encoder.pth')
    torch.save(longnet.state_dict(), longnet_path)
    projector_path = os.path.join(output_dir, 'projector.pth')
    torch.save(projector.state_dict(), projector_path)
    
    return new_lora_path, longnet_path, projector_path
