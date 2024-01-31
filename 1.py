#=======通过下面这个代码可以debug学习chatglm3的代码.
from modeling_chatglm import ChatGLMForConditionalGeneration,ChatGLMConfig
import torch
 
def run ():
  config= ChatGLMConfig(num_layers=2,original_rope=True,use_cache=True)  #=====有一些参数在config.json里面搬过来即可.
  model = ChatGLMForConditionalGeneration(config=config)
  inputs_ids  = torch.randint(low=0,high=config.vocab_size, size=(4,30))
  print(model)
  res = model(inputs_ids)
  print(res)
 
run()