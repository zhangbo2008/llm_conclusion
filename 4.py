#https://github.com/LinkSoul-AI/Chinese-LLaVA 多模态很重要的模型.
# https://hf-mirror.com/LinkSoul/Chinese-LLaVA-Cllama2/blob/main/config.json 知道架构是.   "LlavaLlamaForCausalLM"


from llava import  LlavaLlamaForCausalLM,LlavaConfig

from transformers import AutoConfig
# transformsers 转onnx

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#加上这行之后又恢复以前的速度了!
from PIL import Image
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
if 1:
  config= LlavaConfig.from_json_file(
  'config2.json') 
  # AutoConfig.from_pretained('LinkSoul/Chinese-LLaVA-Cllama2')
  config.num_hidden_layers=2
  llamamodel = LlavaLlamaForCausalLM(config=config) 
  inputs_ids  = torch.randint(low=0,high=config.vocab_size, size=(4,30))
  print(llamamodel)

  a1=CLIPImageProcessor.from_pretrained("conf")
  a=a1.preprocess(load_image('222.png'), return_tensors='pt')['pixel_values'][0].unsqueeze(0)

  res = llamamodel(inputs_ids,images=a)
  print(res)










