import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)
from swift.utils import seed_everything
import torch

MODEL_PATH = '/data/llm-models/qwen2-vl-7b-instruct'

model_type = ModelType.qwen2_vl_7b_instruct
template_type = get_default_template_type(model_type)
model, tokenizer = get_model_tokenizer(
    model_type, torch.bfloat16, model_id_or_path=MODEL_PATH, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 2
template = get_template(template_type, tokenizer)
seed_everything(42)

query = """<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？"""
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

"""
template_type: qwen2-vl
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？
response: 根据图片中的路标，距离各城市的距离如下：

- 马踏：14公里
- 阳江：62公里
- 广州：293公里
query: 距离最远的城市是哪？
response: 距离最远的城市是广州，距离为293公里。
history: [['<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？', '根据图片中的路标，距离各城市的距离如下：\n\n- 马踏：14公里\n- 阳江：62公里\n- 广州：293公里'], ['距离最远的城市是哪？', '距离最远的城市是广州，距离为293公里。']]
"""
