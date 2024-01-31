from transformers.models .llama import LlamaModel,LlamaConfig,LlamaForCausalLM
import torch
 
def run ():
  llamaconfig= LlamaConfig(vocab_size=32000,
  hidden_size=4096//2,
  intermediate_size=1108//2,
  num_hidden_layers=2,
  num_attention_heads=32//2,max_position_embeddings=2048//2)
  llamamodel = LlamaForCausalLM(config=llamaconfig) #https://hf-mirror.com/hiyouga/Llama-2-Chinese-13b-chat/blob/main/config.json 参考这里面的architecture知道llama2依然用的事llamaforcausallm架构.
  inputs_ids  = torch.randint(low=0,high=llamaconfig.vocab_size, size=(4,30))
  print(llamamodel)
  res = llamamodel(inputs_ids)
  print(res)
 
run()