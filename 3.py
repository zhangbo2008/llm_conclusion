from transformers.models .llama import LlamaModel,LlamaConfig,LlamaForCausalLM
import torch
from configuration_phi import  PhiConfig
from modeling_phi import  PhiForCausalLM
def run ():
  config= PhiConfig(
  num_hidden_layers=2,)
  llamamodel = PhiForCausalLM(config=config) #https://hf-mirror.com/hiyouga/Llama-2-Chinese-13b-chat/blob/main/config.json 参考这里面的architecture知道llama2依然用的事llamaforcausallm架构.
  inputs_ids  = torch.randint(low=0,high=config.vocab_size, size=(4,30))
  print(llamamodel)
  res = llamamodel(inputs_ids)
  print(res)
 
run()