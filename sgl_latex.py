pkgs_path = "/bohr/pkgs-7x29/v15/pkgs"
# llava_lib_path = "/bohr/libb-bg5b/v3/llava"
# tsr_model_path = "microsoft/table-structure-recognition-v1.1-all"
model_path = "lmms-lab/llava-onevision-qwen2-7b-si"
cache_path = "/bohr/cach-rxl3/v9/cache"

# pkgs_path = "/personal/pkgs"
# llava_lib_path = "/personal/llava"
# model_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
# cache_path = "/personal/cache"

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

import json
import multiprocessing as mp
import re
from collections import defaultdict
from typing import Optional

from sglang.lang.chat_template import get_chat_template
from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import allocate_init_ports

from sglang import RuntimeEndpoint
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
        # logger.info("Launching server...")
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=launch_server,
            args=(self.server_args, model_overide_args, pipe_writer),
        )
        # logger.info("Waiting for server to launch...")
        proc.start()
        self.pid = proc.pid
        # logger.info("Waiting for server to launch...")
        # pipe_writer.close()
        # timeout = 60
        # import time
        # start_time = time.time()
        #
        # while True:
        #     logger.info("Waiting for initialization state...", flush=True)
        #     if pipe_reader.poll(timeout=1):
        #         logger.info("Waiting for initialization state...", flush=True)
        #         init_state = pipe_reader.recv()
        #         break
        #     if time.time() - start_time > timeout:
        #         raise TimeoutError("Timeout while waiting for initialization state")
        # try:
        #     init_state = pipe_reader.recv()
        # except EOFError:
        #     init_state = ""
        init_state = pipe_reader.recv()

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        self.endpoint = RuntimeEndpoint(self.url)


@sgl.function
def one_image(s, img_path, caption, q3):
    img_path = os.path.join(base_dir, 'test_images', img_path)
    s += sgl.user(
        sgl.image(img_path) +
        f'This is a table image. The caption of the table is "{caption}".')
    s += sgl.assistant("I have a general understanding of the information in this table.")
    s += sgl.user("Convert this table to LaTex.")
    s += sgl.assistant(sgl.gen("LaTex", max_tokens=4096, temperature=0.0, top_p=1))
    s += sgl.user(
        "Based on the provided table, caption and LaTex, select the most relevant subject to the table from (A. Physics, B. Mathematics, C. ComputerScience, D. QuantitativeBiology, E. QuantitativeFinance, F. Statistics, G. ElectricalEngineeringandSystemsScience, H. Economics). Answer with the option's letter from the given choices directly.")
    s += sgl.assistant(
        sgl.gen("subject", choices=["A", "B", "C", "D", "E", "F", "G", "H"],
                max_tokens=2, temperature=0.0, top_p=1))
    s += sgl.user(q3)
    s += sgl.assistant(sgl.gen("option", choices=["A", "B", "C", "D"], max_tokens=2, temperature=0.0, top_p=1))


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


class Worker:

    def __init__(self):
        with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
            self.data = json.load(f)
            self.data = list(self.data)[:10]
        self.batch_size = 4
        self.submission = []

    def run(self):
        model_overide_args = {
            "attn_implementation": "eager",
            "multimodal": True,
            "overwrite_config": {
                "image_aspect_ratio": "anyres_max_9"
            }
        }
        runtime = Runtime(
            model_path=model_path,
            model_overide_args=model_overide_args,
        )
        runtime.endpoint.chat_template = get_chat_template("qwen")
        sgl.set_default_backend(runtime)
        self.process()

    def process(self):
        batch_images = []
        for item in self.data:
            q3 = f"""Based on the provided table, caption and LaTex, for the question: "{item["question"]}", select the most correct option from (A. {item["options"][0]}, B. {item["options"][1]}, C. {item["options"][2]}, D. {item["options"][3]}). Answer with the option\'s letter from the given choices directly."""
            # batch_images.append((item["image_path"], item["caption"], q3))
            batch_images.append({
                "img_path": item["image_path"],
                "caption": item["caption"],
                "q3": q3
            })
            if len(batch_images) == self.batch_size:
                self.batch(batch_images)
                batch_images = []
        with open('submission.json', 'w') as f:
            json.dump(self.submission, f)

    def batch(self, batch_images):
        states = one_image.run_batch(batch_images)
        for i, s in enumerate(states):
            self.clean_out(batch_images[i]["img_path"], s)

    def clean_out(self, img_path, s):
        try:
            latex = s["LaTex"]
            rows, cols = count_rows_cols(latex)
        except:
            rows, cols = -1, -1
        try:
            category = sub_list[l2i[s["subject"][0]]]
        except:
            category = ""
        try:
            answer = l2i[s["option"][0]]
        except:
            answer = -1
        sub_item = {
            "image_path": img_path,
            "category": category,
            "cols": cols,
            "rows": rows,
            "answer": answer,
        }
        print(sub_item)
        self.submission.append(sub_item)


worker = Worker()
worker.run()
