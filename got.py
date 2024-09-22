# -*- coding:utf-8 -*-
# @FileName : tmp.py.py
# @Time : 2024/9/11 15:40
# @Author : fiv


import os
import re

import torch
from PIL import Image
from transformers import AutoTokenizer

from GOT.model import *
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import SeparatorStyle, Conversation
from GOT.utils.utils import KeywordsStoppingCriteria

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
import logging

logger = logging.getLogger(__name__)

import time

filename = str(time.time()) + '.log'
logging.basicConfig(filename=filename, level=logging.INFO)


class GOT:
    def __init__(self, model_name):
        model_name = os.path.expanduser(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = GOTQwenForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            device_map='cuda',
            use_safetensors=True,
            pad_token_id=151643
        )
        self.model.generate
        self.model = torch.compile(self.model)
        self.model.to(device='cuda', dtype=torch.bfloat16)
        self.image_processor = BlipImageEvalProcessor(image_size=1024)
        self.image_processor_high = BlipImageEvalProcessor(image_size=1024)
        image_token_len = 256
        qs = 'OCR with format: '
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs
        conv = Conversation(
            system="""<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.""",
            roles=["<|im_start|>user\n", "<|im_start|>assistant\n"],
            version="mpt",
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        )
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [self.stop_str]

        inputs = self.tokenizer([prompt])
        self.input_ids = torch.as_tensor(inputs.input_ids).cuda()
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

    # def __call__(self, images):
    #     image_tensors = []
    #     image_tensors_1 = []
    #     for image in images:
    #         image_1 = image.copy()
    #         image_tensor = self.image_processor(image)
    #         image_tensor_1 = self.image_processor_high(image_1)
    #         image_tensors.append(image_tensor)
    #         image_tensors_1.append(image_tensor_1)
    #     image_tensors = torch.stack(image_tensors).half().cuda()
    #     image_tensors_1 = torch.stack(image_tensors_1).half().cuda()
    #     # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    #     # input_ids.shape[0] = len(images)
    #     # input_ids = self.input_ids.repeat(len(images), 1)
    #     print("image_tensors.shape", image_tensors.shape)
    #     print("image_tensors_1.shape", image_tensors_1.shape)
    #     input_ids = self.input_ids
    #     print("input_ids.shape", input_ids.shape)
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         output_ids = self.model.generate(
    #             input_ids,
    #             images=[(image_tensors, image_tensors_1)],
    #             do_sample=False,
    #             num_beams=1,
    #             no_repeat_ngram_size=20,
    #             # streamer=streamer,
    #             max_new_tokens=4096,
    #             stopping_criteria=[self.stopping_criteria]
    #         )
    #         outputs_list = []
    #         for output_id in output_ids:
    #             outputs = self.tokenizer.decode(output_id[self.input_ids.shape[1]:]).strip()
    #             if outputs.endswith(self.stop_str):
    #                 outputs = outputs[:-len(self.stop_str)]
    #             outputs = outputs.strip()
    #             outputs_list.append(outputs)
    #         # outputs = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()
    #         #
    #         # if outputs.endswith(self.stop_str):
    #         #     outputs = outputs[:-len(self.stop_str)]
    #         # outputs = outputs.strip()
    #     return outputs_list, [self.count_rows_cols(outputs) for outputs in outputs_list]

    def __call__(self, images):
        image_tensors = []
        for image in images:
            image_1 = image.copy()
            image_tensor = self.image_processor(image).unsqueeze(0).half().cuda()
            image_tensor_1 = self.image_processor_high(image_1).unsqueeze(0).half().cuda()
            image_tensors.append((image_tensor, image_tensor_1))
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        # input_ids.shape[0] = len(images)
        input_ids = self.input_ids.repeat(len(images), 1)
        print("input_ids.shape", input_ids.shape)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                # streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[self.stopping_criteria]
            )
            outputs_list = []
            for output_id in output_ids:
                outputs = self.tokenizer.decode(output_id[self.input_ids.shape[1]:]).strip()
                if outputs.endswith(self.stop_str):
                    outputs = outputs[:-len(self.stop_str)]
                outputs = outputs.strip()
                outputs_list.append(outputs)
            # outputs = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()
            #
            # if outputs.endswith(self.stop_str):
            #     outputs = outputs[:-len(self.stop_str)]
            # outputs = outputs.strip()
        return outputs_list, [self.count_rows_cols(outputs) for outputs in outputs_list]

    # def __call__(self, image):
    #     image_1 = image.copy()
    #     image_tensor = self.image_processor(image).unsqueeze(0).half().cuda()
    #     image_tensor_1 = self.image_processor_high(image_1).unsqueeze(0).half().cuda()
    #     # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         output_ids = self.model.generate(
    #             self.input_ids,
    #             images=[(image_tensor, image_tensor_1)],
    #             do_sample=False,
    #             num_beams=1,
    #             no_repeat_ngram_size=20,
    #             # streamer=streamer,
    #             max_new_tokens=4096,
    #             stopping_criteria=[self.stopping_criteria]
    #         )
    # 
    #         outputs = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()
    # 
    #         if outputs.endswith(self.stop_str):
    #             outputs = outputs[:-len(self.stop_str)]
    #         outputs = outputs.strip()
    #     return outputs, self.count_rows_cols(outputs)

    def count_rows_cols(self, latex_code):
        try:
            # 查找列数：根据表格行的定义找到表格列标识符，如 |l|c|c|c|c|
            columns = re.search(r'\\begin\{tabular\}\{([^\}]+)\}', latex_code)
            if columns:
                num_cols = len([c for c in columns.group(1) if c.isalpha()])
            else:
                num_cols = 0

            # 查找行数：根据 \hline 分隔符统计表格的行数
            rows = latex_code.split(r'\\')
            num_rows = sum(1 for row in rows if '&' in row or '\\rule' in row)

            return num_rows, num_cols
        except:
            return -1, -1


if __name__ == "__main__":
    # model = "model/GOT_weights"
    # model = os.path.expanduser(model)
    # from vllm import LLM
    # llm = LLM(model=model, trust_remote_code=True, dtype="float16")

    got = GOT("model/GOT_weights")
    base_dir = "data/form-recognition-train_v4"
    import json
    import time

    with open(os.path.join(base_dir, "dataset.json"), "r") as f:
        data = json.load(f)
        data = list(data)[:2]
    print(data)
    start = time.time()
    images = [Image.open(os.path.join(base_dir, "test_images", d["image_path"])).convert('RGB')
              for d in data]
    latex, shape = got(images)
    for l, s in zip(latex, shape):
        logger.info(l)
    end = time.time()
    avg = (end - start) / len(data)
    logger.info(f"avg: {avg}")

# 15.6
