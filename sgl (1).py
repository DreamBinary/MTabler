# config env
import json
import logging
import re
from collections import defaultdict
from typing import Optional

from sglang.lang.chat_template import get_chat_template
from sglang.srt.utils import allocate_init_ports

from sglang import RuntimeEndpoint

logging.basicConfig(filename='log.txt', level=logging.INFO)
import multiprocessing as mp
from sglang.srt.server import launch_server

from sglang.srt.server_args import ServerArgs

pkgs_path = "/bohr/pkgs-7x29/v15/pkgs"
# llava_lib_path = "/bohr/libb-bg5b/v3/llava"
# tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"
model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v9/cache"

# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"

# !pip install {pkgs_path}/* --ignore-installed
# !cp {llava_lib_path}. - r

import os

# os.system(f"pip install {pkgs_path}/* --ignore-installed")
# os.system(f"cp -r {llava_lib_path} .")
# # 提交时可能不能联网，设置成离线模式防止联网失败报错
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path
os.environ["HF_HOME"] = cache_path
device = "cuda"

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import warnings
import sglang as sgl
import torch

warnings.filterwarnings("ignore")

l2i = defaultdict(lambda: -1)
for i, letter in enumerate('ABCDEFGH'):
    l2i[letter] = i
sub_list = ('Physics', 'Mathematics', 'ComputerScience', 'QuantitativeBiology', 'QuantitativeFinance',
            'Statistics', 'ElectricalEngineeringandSystemsScience', 'Economics', '')
torch.cuda.empty_cache()
# disable_torch_init()

if os.environ.get('DATA_PATH_B'):  # 提交时会选择隐藏的测试数据集路径（A+B榜），数据集的格式与A榜数据相同，但数目不同（5360张）
    base_dir = os.environ.get('DATA_PATH_B')
else:
    base_dir = '/bohr/form-recognition-train-b6y2/v4'  # 示例，把A榜测试数据集路径作为测试集路径，仅开发时挂载A榜数据用于debug   # 示例，把A榜测试数据集路径作为测试集路径，仅开发时挂载A榜数据用于debug


def count_rows_cols(latex_code):
    try:
        # 查找列数：根据表格行的定义找到表格列标识符，如 |l|c|c|c|c|
        columns = re.search(r'\\begin\{tabular\}\{([^\}]+)\}', latex_code)
        if columns:
            num_cols = len([c for c in columns.group(1) if c.isalpha()])
        else:
            num_cols = 0

        # 查找行数：根据 \hline 分隔符统计表格的行数
        rows = latex_code.split(r'\hline')
        num_rows = sum(1 for row in rows if '&' in row or '\\rule' in row)

        return num_rows, num_cols
    except:
        return -1, -1


@sgl.function
def one_image(s, img_path, caption, q3):
    # output_ids = self.model.generate(
    #     input_ids.cuda(),
    #     images=[img.cuda() for img in image_tensors],
    #     image_sizes=image_sizes,
    #     do_sample=True if args.temperature > 0 else False,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     num_beams=args.num_beams,
    #     max_new_tokens=args.max_new_tokens,
    #     use_cache=True,
    # )
    img_path = os.path.join(base_dir, 'test_images', img_path)
    s += sgl.user(
        sgl.image(img_path) +
        f'This is a table image. This is a table image with red borders. The caption of the table is "{caption}".')
    s += sgl.assistant("I have a general understanding of the information in this table.")
    s += sgl.user("Convert this table to LaTex.")
    s += sgl.assistant(sgl.gen("LaTex", max_tokens=4069, temperature=0.0, top_p=1))
    s += sgl.user(
        "Based on the provided table, caption and LaTex, select the most relevant subject to the table from (A. Physics, B. Mathematics, C. ComputerScience, D. QuantitativeBiology, E. QuantitativeFinance, F. Statistics, G. ElectricalEngineeringandSystemsScience, H. Economics). Answer with the option's letter from the given choices directly.")
    s += sgl.assistant(
        sgl.gen("subject", choices=["A", "B", "C", "D", "E", "F", "G", "H"], max_tokens=2, temperature=0.0,
                top_p=1))
    s += sgl.user(q3)
    s += sgl.assistant(sgl.gen("option", choices=["A", "B", "C", "D"], max_tokens=2, temperature=0.0, top_p=1))


class Runtime(sgl.srt.server.Runtime):
    def __init__(
            self,
            log_level: str = "error",
            model_overide_args: Optional[dict] = None,
            *args,
            **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # Pre-allocate ports
        self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
            self.server_args.port,
            self.server_args.additional_ports,
            self.server_args.dp_size,
        )

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )

        self.pid = None
        logger.info("Launching server...")
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, model_overide_args, pipe_writer),
        )
        proc.start()
        # pipe_writer.close()
        self.pid = proc.pid
        print("Pid:", self.pid)
        init_state = pipe_reader.recv()
        # try:
        #     init_state = pipe_reader.recv()
        # except EOFError:
        #     init_state = ""
        print(init_state)
        print(f"Finish launching server at {self.url}")

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)


class Worker:

    def __init__(self):
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            # self.data = json.load(f)
            self.data = list(json.load(f))[:100]
        self.batch_size = 1
        self.submission = []

    def run(self):
        model_overide_args = {
            "attn_implementation": "eager",
            "multimodal": True,
            "overwrite_config": {
                "image_aspect_ratio": "anyres_max_9"
            }
        }
        print("Building runtime...")
        runtime = Runtime(
            model_path=model_path,
            model_overide_args=model_overide_args,
            show_time_cost=True,
            # disable_flashinfer=True,
            # disable_flashinfer_sampling=True,
            # enable_torch_compile=False,
            # disable_cuda_graph=True,
        )
        print("Setting chat template...")
        runtime.endpoint.chat_template = get_chat_template("qwen")
        sgl.set_default_backend(runtime)
        print("Start processing...")
        # process()
        runtime.shutdown()

    def process(self):
        batch_images = []
        cnt = 0
        for item in self.data:
            print(f"Processing {cnt + 1}/{len(self.data)}")
            path = os.path.join(base_dir, 'test_images', item["image_path"])
            caption = item["caption"]
            q3 = f"""Based on the provided table, caption and LaTex, for the question: "{item["question"]}", select the most correct option from (A. {item["options"][0]}, B. {item["options"][1]}, C. {item["options"][2]}, D. {item["options"][3]}). Answer with the option\'s letter from the given choices directly."""
            batch_images.append((path, caption, q3))
            if len(batch_images) == self.batch_size:
                self.batch(batch_images)
                batch_images = []
        with open('submission.json', 'w') as f:
            json.dump(self.submission, f)

    def batch(self, batch_images):
        states = one_image.run_batch(batch_images)
        for i, s in enumerate(states):
            self.clean_out(batch_images[i][0], s)

    def clean_out(self, img_path, s):
        latex = s["LaTex"]
        rows, cols = count_rows_cols(latex)
        try:
            sub_item = {
                "image_path": img_path,
                "category": sub_list[l2i[s["subject"][0]]],
                "cols": cols,
                "rows": rows,
                "answer": l2i[s["option"][0]],
            }
        except:
            sub_item = {
                "image_path": img_path,
                "category": "",
                "cols": -1,
                "rows": -1,
                "answer": -1,
            }
        print(latex)
        print(sub_item)
        self.submission.append(sub_item)

worker = Worker()
worker.run()
