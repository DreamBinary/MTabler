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
        ).eval()
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

    def __call__(self, image):
        image_1 = image.copy()
        image_tensor = self.image_processor(image)
        image_tensor_1 = self.image_processor_high(image_1)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                self.input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                # streamer=streamer,
                max_new_tokens=4096,
                stopping_criteria=[self.stopping_criteria]
            )

            outputs = self.tokenizer.decode(output_ids[0, self.input_ids.shape[1]:]).strip()

            if outputs.endswith(self.stop_str):
                outputs = outputs[:-len(self.stop_str)]
            outputs = outputs.strip()
        return outputs, self.count_rows_cols(outputs)

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
    got = GOT("../../GOT_weights")
    latex, shape = got(Image.open("../../img/table.png").convert('RGB'))
    print(latex)
    print(shape)
