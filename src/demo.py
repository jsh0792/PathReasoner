import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchscale.model.LongNet import make_longnet_from_name
from modules import ProjectorConfig, ProjectorModel, dispatch_modules
from model import LLaVAModel
from peft import PeftModel, LoraConfig
from xtuner.model.utils import LoadWoInit, find_all_linear_names, traverse_dict

from xtuner.registry import BUILDER

import argparse
import os
import os.path as osp
from types import FunctionType
import deepspeed

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from sympy import im

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import MAP_FUNC
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)
import torch
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from xtuner.utils import PROMPT_TEMPLATE
from PIL import Image
import pandas as pd
import numpy as np
from transformers import GenerationConfig, StoppingCriteriaList

import os
import h5py

pretrained_weight = '/path/to/checkpoint/PathReasoner/model.safetensors'
longnet_weight = '/path/to/checkpoint/PathReasoner/LongNet_encoder/model.pth'
projector_weight = '/path/to/checkpoint/PathReasoner/projector/pytorch_model.bin'
test_image_file = '/path/to/test/image.h5'

llm = AutoModelForCausalLM.from_pretrained(
    pretrained_weight,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    cfg.llm_name_or_path,
    trust_remote_code=True,
    encode_special_tokens=True
)

encoder_name = "LongNet_{}_layers_{}_dim".format(2, 512)
LongNet_encoder = make_longnet_from_name(encoder_name)
LongNet_encoder.load_state_dict(torch.load(longnet_weight, weights_only=True))

projector_depth = 2
hidden_size=512
projector_config = ProjectorConfig(
    visual_hidden_size=hidden_size,
    llm_hidden_size=llm.config.hidden_size,
    depth=projector_depth
)    
projector = ProjectorModel(projector_config)
projector.load_state_dict(torch.load(projector_weight, weights_only=True))

llm.cuda().eval()
LongNet_encoder = LongNet_encoder.to(torch.float16).cuda().eval()
projector = projector.to(torch.float16).cuda().eval()

CoT_prompt = ("You are an expert AI pathologist. Carefully analyze the provided whole slide image (WSI) "
"to answer the following question.\n"
"Follow these steps:\n"
"1. Observe and describe key histopathological findings relevant to the question.\n"
"2. Perform step-by-step clinical reasoning to connect findings with your conclusion.\n"
"3. Provide the final concise answer.\n"
"Please respond in the exact format below:\n"
"<info>\nHistopathological Findings: ...\n</info>\n\n"
"<think>\nClinical Reasoning: ...\n</think>\n\n"
"<answer>\nFinal Answer: ...\n</answer>\n\n"
"Question:\n")

sample_num = 12800

if test_image_file.endswith('.h5'):     
    data = h5py.File(test_image_file, 'r')
    image = data['features'][:]
    image = torch.torch.from_numpy(image)
    total_rows = image.shape[0]
    image = image.reshape(1, image.shape[0], image.shape[1])
else:
    raise ValueError('Not support this format.')
  
image = image.cuda()
prompt_template = PROMPT_TEMPLATE.qwen_chat
SYSTEM = ''
sample_input = 'What is the diagnosis of this WSI?'
sample_input = vqa_CoT_prompt + sample_input

print('Input: ', sample_input)

instruction = prompt_template.get('INSTRUCTION', '{input}')
sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
inputs = (SYSTEM + instruction).format(input=sample_input, round=1)
chunk_encode = []
for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
    if idx == 0:
        cur_encode = tokenizer.encode(chunk)
    else:
        cur_encode = tokenizer.encode(
            chunk, add_special_tokens=False)
    chunk_encode.append(cur_encode)
assert len(chunk_encode) == 2
input_ids = []
for idx, cur_chunk_encode in enumerate(chunk_encode):
    input_ids.extend(cur_chunk_encode)
    if idx != len(chunk_encode) - 1:
        input_ids.append(IMAGE_TOKEN_INDEX)
input_ids = torch.tensor(input_ids).cuda()

image = LongNet_encoder(src_tokens=None, token_embeddings=image.permute.to(torch.float16))["encoder_out"]
pixel_values = projector(image)
mm_inputs = prepare_inputs_labels_for_multimodal(
    llm=llm,
    input_ids=input_ids.unsqueeze(0),
    pixel_values=pixel_values)

max_new_tokens=1280
gen_config = GenerationConfig(
    max_new_tokens=max_new_tokens,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
)
stop_words=[]
stop_words += prompt_template.get('STOP_WORDS', [])
stop_criteria = StoppingCriteriaList()
for word in stop_words:
    stop_criteria.append(
        StopWordStoppingCriteria(tokenizer, word))

generate_output = llm.generate(
    **mm_inputs,
    generation_config=gen_config,
    streamer=None,
    bos_token_id=tokenizer.bos_token_id,
    stopping_criteria=stop_criteria)

generation_output = tokenizer.decode(generate_output[0])
if generation_output.endswith('<|im_end|>'):
    generation_output = generation_output[:-10]
    
print('Output: ', generation_output)