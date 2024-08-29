---
license: apache-2.0
datasets:
- lmms-lab/LLaVA-OneVision-Data
language:
- en
- zh
metrics:
- accuracy
library_name: transformers
tags:
- multimodal

model-index:
- name: llava-onevision-qwen-0.5b-ov
  results:
  - task:
      type: multimodal
    dataset:
      type: ai2d
      name: AI2D
    metrics:
    - name: accuracy
      type: accuracy
      value: 57.1
      verified: true
  - task:
      type: multimodal
    dataset:
      type: chartqa
      name: ChartQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 61.4
      verified: true
  - task:
      type: multimodal
    dataset:
      type: docvqa
      name: DocVQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 73.7
      verified: true
  - task:
      type: multimodal
    dataset:
      type: infovqa
      name: InfoVQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 46.3
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mathverse
      name: MathVerse
    metrics:
    - name: accuracy
      type: accuracy
      value: 17.9
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mathvista
      name: MathVista
    metrics:
    - name: accuracy
      type: accuracy
      value: 34.8
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mmbench
      name: MMBench
    metrics:
    - name: accuracy
      type: accuracy
      value: 52.1
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mme-perception
      name: MME-Perception
    metrics:
    - name: score
      type: score
      value: 1238
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mme-cognition
      name: MME-Cognition
    metrics:
    - name: score
      type: score
      value: 240
      verified: true      
  - task:
      type: multimodal
    dataset:
      type: mmmu
      name: MMMU
    metrics:
    - name: accuracy
      type: accuracy
      value: 31.4
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mmvet
      name: MMVet
    metrics:
    - name: accuracy
      type: accuracy
      value: 29.1
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mmstar
      name: MMStar
    metrics:
    - name: accuracy
      type: accuracy
      value: 37.5
      verified: true
  - task:
      type: multimodal
    dataset:
      type: seed-bench
      name: Seed-Bench
    metrics:
    - name: accuracy
      type: accuracy
      value: 65.5
      verified: true
  - task:
      type: multimodal
    dataset:
      type: science-qa
      name: Science-QA
    metrics:
    - name: accuracy
      type: accuracy
      value: 67.2
      verified: true
  - task:
      type: multimodal
    dataset:
      type: imagedc
      name: ImageDC
    metrics:
    - name: accuracy
      type: accuracy
      value: 83.3
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mmlbench
      name: MMLBench
    metrics:
    - name: accuracy
      type: accuracy
      value: 49.9
      verified: true
  - task:
      type: multimodal
    dataset:
      type: realworldqa
      name: RealWorldQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 55.6
      verified: true
  - task:
      type: multimodal
    dataset:
      type: vibe-eval
      name: Vibe-Eval
    metrics:
    - name: accuracy
      type: accuracy
      value: 33.8
      verified: true
  - task:
      type: multimodal
    dataset:
      type: llava-w
      name: LLaVA-W
    metrics:
    - name: accuracy
      type: accuracy
      value: 74.2
      verified: true
  - task:
      type: multimodal
    dataset:
      type: l-wilder
      name: L-Wilder
    metrics:
    - name: accuracy
      type: accuracy
      value: 55.0
      verified: true
  - task:
      type: multimodal
    dataset:
      type: actnet-qa
      name: ActNet-QA
    metrics:
    - name: accuracy
      type: accuracy
      value: 50.5
      verified: true
  - task:
      type: multimodal
    dataset:
      type: egoschema
      name: EgoSchema
    metrics:
    - name: accuracy
      type: accuracy
      value: 26.8
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mlvu
      name: MLVU
    metrics:
    - name: accuracy
      type: accuracy
      value: 50.3
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mvbench
      name: MVBench
    metrics:
    - name: accuracy
      type: accuracy
      value: 45.5
      verified: true
  - task:
      type: multimodal
    dataset:
      type: nextqa
      name: NextQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 57.2
      verified: true
  - task:
      type: multimodal
    dataset:
      type: percepTest
      name: PercepTest
    metrics:
    - name: accuracy
      type: accuracy
      value: 49.2
      verified: true
  - task:
      type: multimodal
    dataset:
      type: seedbench
      name: SeedBench
    metrics:
    - name: accuracy
      type: accuracy
      value: 44.2
      verified: true
  - task:
      type: multimodal
    dataset:
      type: videochatgpt
      name: VideoChatGPT
    metrics:
    - name: score
      type: score
      value: 3.12
      verified: true
  - task:
      type: multimodal
    dataset:
      type: videodc
      name: VideoDC
    metrics:
    - name: score
      type: score
      value: 3.55
      verified: true
  - task:
      type: multimodal
    dataset:
      type: videomme
      name: VideoMME
    metrics:
    - name: accuracy
      type: accuracy
      value: 44.0
      verified: true
  - task:
      type: multimodal
    dataset:
      type: iei
      name: Image Edit Instruction
    metrics:
    - name: accuracy
      type: accuracy
      value: 17.1
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mi-vqa
      name: MI-VQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 48.7
      verified: true
  - task:
      type: multimodal
    dataset:
      type: nlvr2
      name: NLVR2
    metrics:
    - name: accuracy
      type: accuracy
      value: 63.4
      verified: true
  - task:
      type: multimodal
    dataset:
      type: puzzle
      name: Puzzle
    metrics:
    - name: accuracy
      type: accuracy
      value: 35.4
      verified: true
  - task:
      type: multimodal
    dataset:
      type: q-bench
      name: Q-Bench
    metrics:
    - name: accuracy
      type: accuracy
      value: 48.8
      verified: true
  - task:
      type: multimodal
    dataset:
      type: spot-diff
      name: Spot-Diff
    metrics:
    - name: accuracy
      type: accuracy
      value: 36.4
      verified: true
  - task:
      type: multimodal
    dataset:
      type: tr-vqa
      name: TR-VQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 65.0
      verified: true
  - task:
      type: multimodal
    dataset:
      type: vst
      name: VST
    metrics:
    - name: accuracy
      type: accuracy
      value: 29.8
      verified: true
  - task:
      type: multimodal
    dataset:
      type: scannet-chat
      name: ScanNet-Chat
    metrics:
    - name: accuracy
      type: accuracy
      value: 60.00
      verified: true
  - task:
      type: multimodal
    dataset:
      type: scannet-td
      name: ScanNet-TD
    metrics:
    - name: accuracy
      type: accuracy
      value: 48.00
      verified: true
  - task:
      type: multimodal
    dataset:
      type: scanqa
      name: ScanQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 29.40
      verified: true
  - task:
      type: multimodal
    dataset:
      type: alfred
      name: ALFRED
    metrics:
    - name: accuracy
      type: accuracy
      value: 62.20
      verified: true
  - task:
      type: multimodal
    dataset:
      type: nuscenesvqa
      name: nuScenesVQA
    metrics:
    - name: accuracy
      type: accuracy
      value: 70.50
      verified: true
  - task:
      type: multimodal
    dataset:
      type: blink
      name: BLINK
    metrics:
    - name: accuracy
      type: accuracy
      value: 52.1
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mantis
      name: Mantis
    metrics:
    - name: accuracy
      type: accuracy
      value: 39.6
      verified: true
  - task:
      type: multimodal
    dataset:
      type: mathverse-mv
      name: MathVerse-mv
    metrics:
    - name: accuracy
      type: accuracy
      value: 60.0
      verified: true
  - task:
      type: multimodal
    dataset:
      type: muirbench
      name: MuirBench
    metrics:
    - name: accuracy
      type: accuracy
      value: 25.5
      verified: true
  - task:
      type: multimodal
    dataset:
      type: sciverse-mv
      name: SciVerse-mv
    metrics:
    - name: accuracy
      type: accuracy
      value: 29.1
      verified: true      
---


# LLaVA-OneVision

![banner](https://i.postimg.cc/pL17YtG4/WX20240508-220230-2x.png)

Play with the model on the [LLaVA OneVision Chat](https://llava-onevision.lmms-lab.com/).

##  Table of Contents

1. [Model Summary](##model-summary)
2. [Use](##use)
3. [Limitations](##limitations)
4. [Training](##training)
5. [License](##license)
6. [Citation](##citation)

## Model Summary

The LLaVA-OneVision models are 0.5/7/72B parameter models trained on [LLaVA-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data), based on Qwen2 language model with a context window of 32K tokens.

- **Repository:** [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT?tab=readme-ov-file)
- **Project Website:** [llava-onevision.lmms-lab.com](llava-onevision.lmms-lab.com)
- **Paper:** [LLaVA-OneVision]()
- **Point of Contact:** [Bo Li](mailto:drluodian@gmail.com)
- **Languages:** English, Chinese


## Use

### Intended use

The model was trained on [LLaVA-OneVision Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) and have the ability to interact with images, multi-image and videos. 

**Feel free to share your generations in the Community tab!**

### Generation

We provide the simple generation process for using our model. For more details, you could refer to [Github](https://github.com/LLaVA-VL/LLaVA-NeXT).

```python
# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava_cp.model.builder import load_pretrained_model
from llava_cp.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava_cp.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX
from llava_cp.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                      device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
    device)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
```

# Training

## Model

- **Architecture:** SO400M + Qwen2
- **Pretraining Stage:** LCS-558K, 1 epoch, projector
- **Mid Stage:** A mixture of 4.7M high-quality synthetic data, 1 epoch, full model
- **Final-Image Stage:** A mixture of 3.6M single-image data, 1 epoch, full model
- **OneVision Stage:** A mixture of 1.6M single-image/multi-image/video data, 1 epoch, full model
- **Precision:** bfloat16

## Hardware & Software

- **GPUs:** 256 * Nvidia Tesla A100 (for whole model series training)
- **Orchestration:** [Huggingface Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- **Neural networks:** [PyTorch](https://github.com/pytorch/pytorch)

# Citation
```
@article{li2024llavaonevision,
      title={LLaVA-OneVision}, 
}
```