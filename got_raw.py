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
        qs = 'OCR a latex table with format: '
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
            print("-->>output_ids.shape", output_ids.shape)

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
    got = GOT("model/GOT_weights")
    base_dir = "data/form-recognition-train_v4"
    import json
    import time

    with open(os.path.join(base_dir, "dataset.json"), "r") as f:
        data = json.load(f)
        data = list(data)[:10]
    start = time.time()
    images = [Image.open(os.path.join(base_dir, "test_images", d["image_path"])).convert('RGB')
              for d in data]
    for image in images:
        got(image)
        latex, shape = got(image)
        print(latex)
    end = time.time()
    avg = (end - start) / len(data)


s = """
/root/anaconda3/bin/conda run -n llava --no-capture-output python /root/MTabler/got_raw.py 
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` model input instead.
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{|l|ccc|c|}
\hline \begin{tabular}{l} 
Relaxation \\
constraints
\end{tabular} & \begin{tabular}{c} 
Avg. \\
solving \\
time (s)
\end{tabular} & \begin{tabular}{c} 
Max. \\
solving \\
time (s)
\end{tabular} & \begin{aligned} Solved instances (in \% of total \\
number of instances) \\
with preimages \end{aligned} & \\
\hline\(\rho_{1}\) & 12 & 80 & 65 & 35 \\
\hline\(\rho_{2}\) & 46 & 250 & 75 & 25 \\
\hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{llr}
\hline Method & Key Point & Accuracy \\
\hline Wu et al. & Feature learning from 3D voxelized input & 77.3 \\
Wu et al. & Unsupervised feature learning using 3D generative adversarial modeling & 83.3 \\
Qi et al. & Point-cloud based representation learning & 86.2 \\
Su et al. & Multi-view 2D images for feature learning & 90.1 \\
Qi et al. & Volumetric and multi-view feature learning & 91.4 \\
Brock et al. & Generative and discriminative voxel modeling & 95.5 \\
\hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{lcccccc} 
Model & \(N\) & \(L\) & \(\operatorname{dim}\) & \((\mathrm{s})\) & \((\mathrm{s})\) & \((\mathrm{s})\) & \((\) s \()\) \\
\hline Random & 5 & \(\leq 1\) & 0.00 & 0.00 & 0.05 & 0.71 \\
Random & 10 & \(\leq 1\) & 0.00 & 0.0 & 0.93 & 12.31 \\
Random & 10 & \(\leq 2\) & 0.01 & 0.00 & 26.60 & \(>2 \mathrm{~h}\) \\
Random & 15 & \(\leq 1\) & 0.00 & 0. & 5.01 & 41.31 \\
Random & 15 & \(\leq 2\) & 0.01 & 0.0 & 61.98 & \(>2 \mathrm{~h}\) \\
Random & 15 & \(-3\) & 0.04 & 0.02 & 418.22 & \(>2 \mathrm{~h}\) \\
\hline Formula & & 5 & \(\leq 2\) & 0.00 & 0.01 & 0.25 \\
Formula & & 10 & \(\leq 2\) & 0.01 & \(0.02 \quad 1.52\) & \\
Formula & & 10 & \(\leq 3\) & 0.62 & 0.57 & 1.52 \\
Formula & & 25 & \(\leq 2\) & 0.06 & 0.05 & 48.81 \\
Formula & & 25 & \(\leq 3\) & 2.89 & 1.58 & 108.46 \\
Formula & & 25 & \(\leq 5\) & \(>2 \mathrm{~h}\) & 2176.83 & 255.39 \\
\hline FIFO(1) & 4 & 4 & 1 & 0.02 & 0.00 & 0.01 & 0.03 \\
FIFO(2) & 13 & 7 & 2 & 0.01 & 0.01 & 1.03 & 0.24 \\
FIFO(3) & 65 & 11 & 3 & 0.35 & 0.70 & 9.32 & 2.44 \\
FIFO(4) & 440 & 16 & 4 & 37.60 & 39.77 & 68.44 & 15.33 \\
FIFO(5) & 3686 & 22 & 5 & \(>2 \mathrm{~h}\) & 3027.54 & 382.33 & 71.59 \\
\hline ww(1) & 4 & 4 & 1 & 0.02 & \(0.00 \quad 0.01\) & 0.03 \\
ww(2) & 8 & 6 & 2 & 0.00 & 0.00 & 0.12 & 0.03 \\
ww(3) & 24 & 8 & 3 & 0.16 & 0.17 & 0.75 & 0.16 \\
ww(4) & 112 & 10 & 4 & 23.71 & 30.51 & 2.85 & 0.60 \\
ww(5) & 728 & 12 & 5 & 5880.04 & \(>2 \mathrm{~h}\) & 9.27 & 1.83 \\
\hline & 5 & 2 & 0.02 & 0.00 & 1.08 & 0.06 \\
& 5 & 2 & 0.01 & 0.00 & 1.71 & 0.04
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{|l|r|r|r|r|}
\hline & Mod-OMP DaF & OMP DaF & Mod-SOMP DeF & SOMP DeF \\
\hline\(a\) & 9.966 & 9.982 & 10 & 10 \\
\hline\(b\) & 0.034 & 0.018 & 0 & 0 \\
\hline\(c\) & 0.842 & 2.906 & 5.194 & 11.944 \\
\hline\(d\) & 30.158 & 28.094 & 25.806 & 19.056 \\
\hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{|c|c|c|c|}
\hline Specs & \begin{tabular}{c} 
XSEDE \\
Stampede-Gordon
\end{tabular} & \begin{tabular}{c} 
DIDCLAB \\
WS1-WS-2
\end{tabular} & EC2 \\
\hline Bandwidth (Gbps) & 10 & 1 & 10 \\
\hline RTT (ms) & 40 & 0.2 & 100 \\
\hline TCP Buffer Size (MB) & 32 & 4 & 60 \\
\hline BDP (MB) & 48 & 0.02 & 125 \\
\hline File System & Lustre & NFS & SAN \\
\hline \begin{tabular}{c} 
Max File System \\
Throughput (MB)
\end{tabular} & 1200 & 90 & 320 \\
\hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{|cccc|}
\hline Methods- \(\mathcal{P}\) & 80 & 200 & 450 \\
\hline \hline GA Deployment Method & \(99.2 \%\) & \(88.6 \%\) & \(75.3 \%\) \\
K-Means & \(98.6 \%\) & \(82.3 \%\) & \(69.4 \%\) \\
Branch and Cut & \(95.6 \%\) & \(83.1 \%\) & \(69.4 \%\) \\
Greedy Search & \(92.6 \%\) & \(79.1 \%\) & \(71.4 \%\) \\
Random & \(85.6 \%\) & \(72.1 \%\) & \(59.4 \%\) \\
\hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{lcccccc}
\hline Method & backbone & PQ & \multirow{2}{*}{ PQ \(^{\text {Th }}\)} & \multirow{2}{*}{ PQ \(^{\text {St }}\)} & AP & mIoU \\
\hline DWT & VGG16 & - & - & - & 21.2 & - \\
SGN & VGG16 & - & - & - & 29.2 & - \\
Li et. al. & ResNet-101 & 53.8 & 42.5 & 62.1 & 28.6 & - \\
\hline Mask R-CNN & ResNet-50 & - & - & - & 31.5 & - \\
\hline Ours & ResNet-50-FPN & 55.0 & 51.2 & 57.8 & 32.2 & - \\
Ours & ResNet-50-FPN & 56.4 & 52.7 & 59.0 & 33.6 & 73.6 \\
Ours & ResNet-101-FPN & \(\mathbf{5 9 . 0}\) & \(\mathbf{5 4 . 8}\) & \(\mathbf{6 2 . 1}\) & \(\mathbf{3 4 . 4}\) & \(\mathbf{7 5 . 6}\) \\
\hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{lrrrrr}
\hline Backbone & Grouping & Dynamic weighting & mIoU & mAcc & aAcc \\
\hline PSANet & & & 46.65 & 53.2 & 89.04 \\
PSANet & & \(\checkmark\) & 50.35 & 66.74 & 87.48 \\
\hline PSANet & \(\checkmark\) & & 60.32 & 64.32 & 91.52 \\
PSANet & \(\checkmark\) & \(\checkmark\) & 79.26 & 89.43 & 93.47 \\
\hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{lllll}
\hline \hline Features & 2D Network & 2D Network & 3D Network & ComboNet-2D/3D \\
\hline Input size & 512 & 256 & \(128 \times 20\) & \((512,128) \times 20\) \\
Number of Blocks & 6 & 5 & 4 & 6,4 \\
Number of Conv. layers per Block & 3 & 3 & 3 & 3 \\
Feature Scale & 8 & 4 & 2 & 8,2 \\
Central block Size & \(8 \times 8 \times 256\) & \(8 \times 8 \times 256\) & \(8 \times \times 256\) & \(8 \times 8 \times 250,8 \times 8 \times 256\) \\
Number of parameters & 13818297 & 13811569 & 35756321 & 49575473*** \\
Speed (seconds for 250 slices)* & 0.08 & 0.07225 & 0.09125 & 0.14275 \\
\hline \hline
\end{tabular}
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
-->> 1
image_features 1
-->>
cur_input_ids 289
cur_input_ids 289
new_input_embeds 1
\begin{tabular}{cccc}
\hline Parameter & Notation & Legal values & Dimension \\
\hline Maximal ship speed & \(v_{s}\) & \(4,6,8,10\) & 4 \\
Thrust speed & \(v_{t}\) & \(1,2,3,4,5\) & 5 \\
Maximal missile speed & \(v_{m}\) & \(1,2,3,4,5,6,7,8,9,10\) & 10 \\
Cooldown time & \(d\) & \(1,2,3,4,5,6,7,8,9\) & 9 \\
Missile cost & \(c\) & \(0,1,5,10,20,50,75,100\) & 8 \\
Ship radius & \(s r\) & \(10,20,30,40,50\) & 5 \\
\hline
\end{tabular}

进程已结束，退出代码为 0


"""