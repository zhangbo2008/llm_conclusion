#https://github.com/LinkSoul-AI/Chinese-LLaVA 多模态很重要的模型.
# https://hf-mirror.com/LinkSoul/Chinese-LLaVA-Cllama2/blob/main/config.json 知道架构是.   "LlavaLlamaForCausalLM"


from llava import  LlavaLlamaForCausalLM,LlavaConfig

from transformers import AutoConfig
# transformsers 转onnx

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#加上这行之后又恢复以前的速度了!

import torch

def run ():
  config= LlavaConfig.from_json_file(
  'config2.json') 
  # AutoConfig.from_pretained('LinkSoul/Chinese-LLaVA-Cllama2')
  config.num_hidden_layers=2
  llamamodel = LlavaLlamaForCausalLM(config=config) 
  inputs_ids  = torch.randint(low=0,high=config.vocab_size, size=(4,30))
  print(llamamodel)
  res = llamamodel(inputs_ids)
  print(res)
 
run()









